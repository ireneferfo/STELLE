import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from time import time
import optuna
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
from abc import ABC

from .model_metrics import weighted_accuracy, sensitivity_specificity
from ..explanations.local_explanation import LocalExplanation


@dataclass
class BaseModelConfig:
    """Base configuration for all model variants."""

    kernel: Any
    concepts: Any
    num_classes: int
    Geps: float = 1e-12
    initial_epsilon: float = math.exp(1)
    initial_concept_relevance_scale: float = 1
    dropout: float = 0.2
    tune: bool = True
    parallel_workers: int = 0
    use_weights: bool = True
    logging: bool = False
    activation: Any = field(default_factory=lambda: nn.GELU())
    epsilon: Optional[float] = None
    hidden_size: int = 512
    num_layers: int = 1
    crel_norm: bool = False
    output_activation: Any = field(default_factory=lambda: nn.Softsign())

class ForwardWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        class_scores, _, _, _ = self.model.forward(x, trainingmode=False)
        return class_scores
            

class BaseConceptModel(nn.Module, ABC):
    """
    Base class for concept-based models.
    
    This abstract base class provides the common infrastructure for all
    concept-based model variants, including training, validation, and
    explanation generation.
    
    Subclasses must implement:
    - _init_variant_parameters: Initialize variant-specific parameters
    - _compute_temporal_embeddings: Compute trajectory embeddings
    - _compute_concept_relevance: Compute concept_relevance weights
    """

    def __init__(self, config: BaseModelConfig):
        super().__init__()

        # Store configuration
        self.config = config
        self.device = config.kernel.device
        self.kernel = config.kernel
        self.normalize = config.kernel.stl_kernel.normalize
        self.concepts = config.concepts
        self.num_classes = config.num_classes
        self.tune = config.tune
        self.parallel_workers = config.parallel_workers
        self.crel_norm = config.crel_norm
        self.weights_flag = config.use_weights
        self.logging = config.logging

        # Initialize common parameters
        self._init_common_parameters(config)
        
        # Initialize variant-specific parameters (hook for subclasses)
        self._init_variant_parameters(config)
        
        # Initialize classifier
        self._init_classifier(config)

        # Initialize state variables
        self._init_state_variables()

        # Validate kernel state
        if config.kernel.phis is not None and len(config.kernel.phis) != len(
            config.concepts
        ):
            config.kernel.phis = None

    def _init_common_parameters(self, config: BaseModelConfig):
        """Initialize parameters common to all variants."""
        # concept_relevance scale
        self.crel_scale = nn.Parameter(
            torch.tensor(math.log(config.initial_concept_relevance_scale), device=self.device),
            requires_grad=config.tune,
        )

        # Epsilon parameter
        self.log_eps = nn.Parameter(
            torch.log(torch.tensor(config.initial_epsilon, device=self.device)),
            requires_grad=config.tune,
        )

        # Geps parameter
        if config.tune:
            self.Geps = nn.Parameter(
                torch.tensor(
                    [config.Geps] * config.num_classes, device=self.device
                ).log()
            )
        else:
            self.Geps = torch.tensor(
                [config.Geps] * config.num_classes, device=self.device
            )

        # Fixed epsilon if provided
        self.epsilon = torch.tensor(config.epsilon) if config.epsilon else None

        # Regularization parameters
        self.crel_collapse_strength = nn.Parameter(
            torch.tensor(-6.0), requires_grad=config.tune
        )
        self.eps_control_strength = nn.Parameter(torch.tensor(-1.0))
        self.log_sigmoid_temp = nn.Parameter(
            torch.log(torch.tensor(1.0)), requires_grad=config.tune
        )

    def _init_variant_parameters(self, config: BaseModelConfig):
        """Initialize variant-specific parameters. Must be implemented by subclasses."""
        pass

    def _init_classifier(self, config: BaseModelConfig):
        """Initialize the classifier network."""
        self.dropout = nn.Dropout(config.dropout)
        self.output_activation = config.output_activation.to(self.device)

        input_dim = len(config.concepts) * config.num_classes

        if config.num_layers == 0:
            self.classifier = nn.Linear(input_dim, config.num_classes).to(self.device)
        else:
            layers = []
            current_dim = input_dim

            for _ in range(config.num_layers):
                layers.extend(
                    [
                        nn.Linear(current_dim, config.hidden_size),
                        nn.LayerNorm(config.hidden_size),
                        config.activation,
                        nn.Dropout(config.dropout),
                    ]
                )
                current_dim = config.hidden_size

            layers.extend(
                [
                    nn.Linear(current_dim, input_dim),
                    nn.Linear(input_dim, config.num_classes),
                ]
            )

            self.classifier = nn.Sequential(*layers).to(self.device)

    def _init_state_variables(self):
        """Initialize state tracking variables."""
        self.kpca = None
        self.robs_mean = None
        self.robs_std = None
        self.robs_dict = None
        self.temb_dict = None
        self.epoch_times = []
        self.weights = None

    def update_parameters(
        self,
        initial_epsilon: Optional[float] = None,
        initial_concept_relevance_scale: Optional[float] = None,
    ):
        """Update model initialization parameters."""
        if initial_epsilon is not None:
            self.log_eps = nn.Parameter(
                torch.log(torch.tensor(initial_epsilon, device=self.device)),
                requires_grad=self.tune,
            )

        if initial_concept_relevance_scale is not None:
            self.crel_scale = nn.Parameter(
                torch.tensor(math.log(initial_concept_relevance_scale), device=self.device),
                requires_grad=self.tune,
            )

    # ========================================================================
    # Forward pass
    # ========================================================================

    def forward(
        self, x: torch.Tensor, trainingmode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input trajectories
            trainingmode: Whether in training mode (uses cached embeddings)

        Returns:
            Tuple of (class_scores, concept_relevance, temb, crelG_raw, G_phis)
        """
        x = x.to(self.device)

        # Compute temporal embeddings (variant-specific)
        temb = self._compute_temporal_embeddings(x, trainingmode)

        # Compute concept_relevance (variant-specific)
        concept_relevance = self._compute_concept_relevance(temb) # (batch, phis)

        # Apply dropout
        concept_relevance_dropped = self.dropout(concept_relevance)

        # Compute G_phis based on robustness values
        G_phis = self._compute_G_phis_matrix(x, concept_relevance, trainingmode) # (batch, phis, classes)
        # Compute concept_relevance-weighted features
        crelG_raw = self._compute_weighted_features(concept_relevance_dropped, G_phis)

        # Apply output activation and classify
        crelG = self.output_activation(crelG_raw).to(self.device) # (batch, phis * classes)
        class_scores = self.classifier(crelG.float().to(self.device))
        if trainingmode:
            return class_scores
        return class_scores, concept_relevance, crelG_raw, G_phis

    def _compute_temporal_embeddings(
        self, x: torch.Tensor, trainingmode: bool
    ) -> torch.Tensor:
        if trainingmode:
            # Use cached embeddings during training
            temb = torch.stack(
                [self.temb_dict[tuple(map(tuple, traj.tolist()))] for traj in x]
            ).to(self.device)
        else:
            # Compute embeddings on the fly during inference
            epsilon = self.epsilon if self.epsilon else F.softplus(self.log_eps)
            with torch.no_grad():
                temb = (
                    self.kernel.compute_rho_phi(
                        x,
                        self.concepts,
                        epsilon,
                        save=True,
                        parallel_workers=self.parallel_workers,
                    )
                    .to(self.device, non_blocking=True)
                    .float()
                )

        return temb

    def _compute_concept_relevance(self, temb: torch.Tensor) -> torch.Tensor:
        """Compute concept_relevance weights from temporal embeddings."""
        # Normalize temporal embeddings
        temb_normalized = (temb - temb.mean()) / (temb.std() + 1e-8)
        temb_normalized = torch.clamp(temb_normalized, min=-10.0, max=10.0)

        # Compute concept_relevance
        concept_relevance = compute_concept_relevance_jit(
            temb_normalized, self.crel_scale, self.crel_norm
        ).to(self.device)

        return concept_relevance

    # ========================================================================
    # G_phis computation
    # ========================================================================

    def _compute_G_phis_matrix(
        self, x: torch.Tensor, concept_relevance: torch.Tensor, trainingmode: bool
    ) -> torch.Tensor:
        """Compute G_phis matrix based on configuration."""        
        if trainingmode:
            return torch.stack(
                [
                    self._compute_G_phis_from_robs(
                        self.robs_dict.get(tuple(map(tuple, traj.tolist())))
                    )
                    for traj in x
                ]
            ).to(self.device, non_blocking=True)
        else:
            with torch.no_grad():
                robs = self._compute_selective_robustness(x, concept_relevance)
                return self._compute_G_phis_from_robs(robs).to(self.device)

    def _compute_selective_robustness(
        self, x: torch.Tensor, concept_relevance: torch.Tensor
    ) -> torch.Tensor:
        """Compute robustness values only for high-concept_relevance concepts."""
        crel_threshold = torch.quantile(concept_relevance, 0.1, dim=1, keepdim=True)
        crel_mask = concept_relevance > crel_threshold

        robs = torch.zeros(x.size(0), len(self.concepts), device=self.device)

        for idx, phi in enumerate(self.concepts):
            if crel_mask[:, idx].any():
                robs[:, idx] = (
                    phi.quantitative(
                        x,
                        evaluate_at_all_times=False,
                        vectorize=False,
                        normalize=self.normalize,
                    )
                    .squeeze()
                    .to(self.device, non_blocking=True)
                )

        return robs

    def _compute_G_phis_from_robs(self, robs: torch.Tensor) -> torch.Tensor:
        """Transform robustness values using class statistics."""
        Geps = torch.exp(self.Geps) if self.tune else self.Geps
        # invstd = 1 / (self.robs_std.cpu() + Geps.unsqueeze(-1).cpu()).to(self.device)
        invstd = 1 / (self.robs_std + Geps.unsqueeze(-1))
        
        return torch.stack(
            [
                torch.abs(robs.to(self.device) - self.robs_mean[c].to(self.device))
                * invstd[c]
                for c in range(self.num_classes)
            ],
            dim=-1,
        )

    def _compute_weighted_features(
        self, concept_relevance: torch.Tensor, G_phis: torch.Tensor
    ) -> torch.Tensor:
        """Compute concept_relevance-weighted features."""
        detached_concept_relevance = concept_relevance.detach()
        crelG_raw = (
            (detached_concept_relevance.unsqueeze(-1) * G_phis)
            .reshape(G_phis.size(0), -1)
            .to(self.device)
        )
        return crelG_raw

    # ========================================================================
    # Training
    # ========================================================================

    def train_model(
        self,
        optimizer,
        trainloader,
        valloader: Optional[Any] = None,
        num_epochs: int = 50,
        verbose: int = 0,
        patience: Optional[int] = None,
        get_robs: bool = False,
        robs: Optional[torch.Tensor] = None,
        checkpath: Optional[str] = None,
        check_frequency: Optional[int] = None,
        val_every_n_epochs: int = 1,
        trial: Optional[optuna.Trial] = None,
    ):
        """Train the model with optional validation and early stopping."""
        start_time = time()
        self._initialize_training(trainloader)

        if check_frequency is None:
            check_frequency = num_epochs // 10 + 1

        checkpoint_epoch, checkpoint_time = self._load_or_compute_robustness(
            trainloader, robs, checkpath
        )

        best_state = self._training_loop(
            optimizer=optimizer,
            trainloader=trainloader,
            valloader=valloader,
            num_epochs=num_epochs,
            start_epoch=checkpoint_epoch,
            start_time=checkpoint_time,
            verbose=verbose,
            patience=patience,
            checkpath=checkpath,
            check_frequency=check_frequency,
            val_every_n_epochs=val_every_n_epochs,
            trial=trial,
        )

        total_time = time() - start_time + checkpoint_time

        if get_robs:
            return robs

        return best_state["train_acc"], best_state["best_epoch"], total_time

    def _initialize_training(self, trainloader):
        """Initialize training state."""
        class_counts = trainloader.dataset.class_distribution.to(torch.float32)
        self.weights = (
            (1.0 / class_counts).to(self.device) if self.weights_flag else None
        )

    def _load_or_compute_robustness(
        self, trainloader, robs: Optional[torch.Tensor], checkpath: Optional[str]
    ) -> Tuple[int, float]:
        """Load checkpoint or compute robustness information."""
        if checkpath and os.path.exists(checkpath):
            pass # return self._load_checkpoint(checkpath)

        if self.robs_mean is None: # if self.Gx and ...
            self._compute_and_store_robustness(trainloader, robs)

        return 0, 0

    def _compute_and_store_robustness(self, trainloader, robs: Optional[torch.Tensor]):
        """Compute and store robustness statistics."""
        if self.logging:
            print("Computing robustness information")

        if robs is None:
            robs = (
                torch.stack(
                    [
                        phi.quantitative(
                            trainloader.dataset.trajectories,
                            evaluate_at_all_times=False,
                            vectorize=False,
                            normalize=self.normalize,
                        ).to(self.device)
                        for phi in self.concepts
                    ],
                    dim=1,
                )
                .squeeze()
                .to(self.device)
            )

        robs_mean = torch.empty(self.num_classes, robs.size(1), device=self.device)
        robs_std = torch.empty(self.num_classes, robs.size(1), device=self.device)

        for cls in range(self.num_classes):
            robs_not_cls = robs[trainloader.dataset.labels != cls]
            robs_mean[cls] = torch.mean(robs_not_cls, dim=0)
            robs_std[cls] = torch.std(robs_not_cls, dim=0)

        self.robs_dict = {
            tuple(map(tuple, traj.tolist())): robs[i]
            for i, traj in enumerate(trainloader.dataset.trajectories)
        }

        self.set_robustness_info(robs_mean, robs_std)

    def _training_loop(
        self,
        optimizer,
        trainloader,
        valloader,
        num_epochs: int,
        start_epoch: int,
        start_time: float,
        verbose: int,
        patience: Optional[int],
        checkpath: Optional[str],
        check_frequency: int,
        val_every_n_epochs: int,
        trial: Optional[optuna.Trial],
    ) -> Dict[str, Any]:
        """Main training loop."""
        self.train()

        best_state = {
            "train_acc": [],
            "best_val_loss": float("inf"),
            "patience_counter": 0,
            "best_model_state": None,
            "best_epoch": 0,
        }

        last_epsilon = None

        for epoch in range(start_epoch, num_epochs):
            epoch_start = time()

            last_epsilon = self._update_temporal_embeddings(trainloader, last_epsilon)
            train_metrics = self._train_epoch(trainloader, optimizer)

            self.epoch_times.append(time() - epoch_start)
            best_state["train_acc"].append(train_metrics["weighted_acc"])

            if self.logging:
                self._log_training_metrics(train_metrics, epoch)

            if self._should_save_checkpoint(
                epoch, start_epoch, checkpath, check_frequency
            ):
                self._save_checkpoint(
                    checkpath, epoch, time() - start_time + start_time
                )

            if self._should_validate(valloader, epoch, val_every_n_epochs):
                val_metrics = self.validate(valloader)

                if self._update_best_state(val_metrics["avg_valloss"], best_state):
                    best_state["best_model_state"] = self.state_dict()
                    best_state["best_epoch"] = epoch

                if self._check_early_stopping(patience, best_state):
                    if verbose > 0:
                        print("Early stopping triggered")
                    if best_state["best_model_state"]:
                        self.load_state_dict(best_state["best_model_state"])
                    break

                if verbose > 0 and (epoch + 1) % verbose == 0:
                    self._print_epoch_summary(epoch, train_metrics, val_metrics)

                if trial and self._should_prune_trial(
                    trial, epoch, val_metrics["weighted_acc"]
                ):
                    raise optuna.TrialPruned()

            elif verbose > 0 and (epoch + 1) % verbose == 0:
                print(
                    f"Epoch {epoch+1}, Loss: {train_metrics['loss']:.5f}, "
                    f"Accuracy: {train_metrics['weighted_acc']:.2f}%"
                )

            torch.cuda.empty_cache()

        return best_state

    def _update_temporal_embeddings(
        self, trainloader, last_epsilon: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Update temporal embeddings if epsilon changed."""
        epsilon = self.epsilon if self.epsilon else F.softplus(self.log_eps)

        if last_epsilon is None or not torch.allclose(last_epsilon, epsilon):
            self.eval()
            temb_dict = {}

            for batch in trainloader:
                if self.device == 'mps':
                    batch[0] = batch[0].float()
                trajectories = batch[0].to(self.device)
                temb_batch = (
                    self.kernel.compute_rho_phi(
                        trajectories,
                        self.concepts,
                        epsilon,
                        save=True,
                        parallel_workers=self.parallel_workers,
                    )
                    .to(self.device, non_blocking=True)
                    .to(torch.float)
                )

                for i, traj in enumerate(trajectories):
                    temb_dict[tuple(map(tuple, traj.cpu().tolist()))] = temb_batch[i]

            self.temb_dict = temb_dict

            if self.logging:
                temb_std = torch.stack(list(temb_dict.values())).std()
                print(f"Computed temporal embeddings (std: {temb_std.item():.6f})")

            return epsilon

        return last_epsilon

    def _train_epoch(self, trainloader, optimizer) -> Dict[str, Any]:
        """Train for one epoch."""
        total_loss = 0
        all_preds = []
        all_labels = []
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        for trajectories, labels in trainloader:
            trajectories = trajectories.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            class_scores = self(trajectories)
            loss = compute_loss(class_scores, labels, self)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                self.crel_scale.clamp_(min=-10, max=10)

            total_loss += loss.item()
            _, labels_idx = torch.max(labels, 1)
            preds = class_scores.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels_idx.cpu().tolist())

            for t, p in zip(labels_idx, preds):
                class_total[t.item()] += 1
                if t == p:
                    class_correct[t.item()] += 1

            del class_scores, loss, preds

        torch.cuda.empty_cache()

        return {
            "loss": total_loss / len(trainloader),
            "weighted_acc": weighted_accuracy(all_labels, all_preds),
            "class_correct": class_correct,
            "class_total": class_total,
        }

    def _should_save_checkpoint(
        self, epoch: int, start_epoch: int, checkpath: Optional[str], check_frequency: int
    ) -> bool:
        """Determine if checkpoint should be saved."""
        return (
            epoch != start_epoch
            and checkpath is not None
            and epoch % check_frequency == 0
        )

    def _save_checkpoint(self, checkpath: str, epoch: int, elapsed_time: float):
        """Save model checkpoint."""
        if os.path.exists(checkpath):
            os.remove(checkpath)

        try:
            torch.save((self.cpu(), epoch, elapsed_time), checkpath)
            if self.logging:
                print(f"Checkpoint saved at epoch {epoch}")
            self.to(self.device)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self, checkpath: str):
        """Load model checkpoint."""
        if not os.path.exists(checkpath):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpath}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpath, map_location=self.device)
            
            # Unpack the saved data
            model_state, epoch, elapsed_time = checkpoint
            
            # Load the model state
            self.load_state_dict(model_state.state_dict())
            self.to(self.device)
            
            if self.logging:
                print(f"Checkpoint loaded from epoch {epoch}")
                print(f"Elapsed training time: {elapsed_time:.2f}s")
            
            return epoch, elapsed_time
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            raise
    
    def _should_validate(self, valloader, epoch: int, val_every_n_epochs: int) -> bool:
        """Determine if validation should be performed."""
        return valloader is not None and (epoch + 1) % val_every_n_epochs == 0

    def _update_best_state(self, val_loss: float, best_state: Dict[str, Any]) -> bool:
        """Update best model state if validation loss improved."""
        if val_loss < best_state["best_val_loss"]:
            best_state["best_val_loss"] = val_loss
            best_state["patience_counter"] = 0
            return True
        else:
            best_state["patience_counter"] += 1
            return False

    def _check_early_stopping(
        self, patience: Optional[int], best_state: Dict[str, Any]
    ) -> bool:
        """Check if early stopping criterion is met."""
        return patience is not None and best_state["patience_counter"] >= patience

    def _should_prune_trial(
        self, trial: optuna.Trial, epoch: int, val_acc: float
    ) -> bool:
        """Check if Optuna trial should be pruned."""
        trial.report(val_acc, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch} (val={val_acc:.3f})")
            return True
        return False

    def _log_training_metrics(self, train_metrics: Dict[str, Any], epoch: int):
        """Log training metrics."""
        print(f"\n=== Epoch {epoch + 1} ===")
        print("Class-wise Training Accuracy:")
        for cls in sorted(train_metrics["class_total"]):
            acc = (
                100.0
                * train_metrics["class_correct"][cls]
                / train_metrics["class_total"][cls]
                if train_metrics["class_total"][cls]
                else 0.0
            )
            print(f"  Class {cls}: {acc:.2f}%")

        print(f"Collapse strength: {torch.exp(self.crel_collapse_strength).item():.1e}")
        print(f"Epsilon control strength: {torch.exp(self.eps_control_strength).item():.1e}")
        print(f"concept_relevance scale: {torch.exp(self.crel_scale).item():.5f}")
        print(f"Epsilon: {torch.exp(self.log_eps).item():.4f}")
        print(f"Sigmoid temperature: {torch.exp(self.log_sigmoid_temp):.3f}")

    def _print_epoch_summary(
        self, epoch: int, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]
    ):
        """Print epoch summary."""
        print(
            f"Epoch {epoch+1}, "
            f"train loss: {train_metrics['loss']:.5f}, "
            f"train acc: {train_metrics['weighted_acc']:.2f}%, "
            f"val loss: {val_metrics['avg_valloss']:.5f}, "
            f"val acc: {val_metrics['weighted_acc']:.2f}%"
        )

    # ========================================================================
    # Validation
    # ========================================================================
    
    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = False,
        ):

        self.eval()
        
        # Single batch prediction
        with torch.no_grad():
            x = x.to(self.device)
            
            # Forward pass
            class_scores, _, _, _  = self.forward(x, trainingmode=False)
            
            # Get predictions
            if return_probs:
                # Apply softmax to get probabilities
                predictions = torch.softmax(class_scores, dim=1).cpu()
            else:
                # Get class labels
                predictions = class_scores.argmax(dim=1).cpu()
            
            return predictions


    def validate(self, valloader, by_class_stats: bool = False) -> Dict[str, Any]:
        """Validate the model on validation data."""
        self.eval()

        total_correct = 0
        total_loss = 0.0
        all_preds = []
        all_labels = []
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for trajectories, labels in valloader:
                trajectories = trajectories.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                class_scores, _, _, _ = self(trajectories, trainingmode=False)
                loss = compute_loss(class_scores, labels, self)

                _, labels_idx = torch.max(labels, 1)
                preds = class_scores.argmax(dim=1)
                total_correct += (preds == labels_idx).sum().item()
                total_loss += loss.item()

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels_idx.cpu().tolist())

                for t, p in zip(labels_idx, preds):
                    class_total[t.item()] += 1
                    if t == p:
                        class_correct[t.item()] += 1

                del class_scores, loss, preds

        torch.cuda.empty_cache()

        accuracy = 100.0 * total_correct / len(valloader.dataset)
        weighted_acc = weighted_accuracy(all_labels, all_preds)
        sensitivity, specificity = sensitivity_specificity(
            all_labels, all_preds, self.num_classes
        )
        avg_loss = total_loss / len(valloader)

        if self.logging or by_class_stats:
            print("\nClass-wise Validation Accuracy:")
            for cls in sorted(class_total):
                acc = (
                    100.0 * class_correct[cls] / class_total[cls]
                    if class_total[cls]
                    else 0.0
                )
                print(f"  Class {cls}: {acc:.2f}%")

            if self.logging:
                print(f"Learned epsilon: {F.softplus(self.log_eps).item():.6f}")

        return {
            "accuracy": accuracy,
            "weighted_acc": weighted_acc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "avg_valloss": avg_loss,
        }

    # ========================================================================
    # Explanation methods
    # ========================================================================

    def get_explanations(
        self,
        x: torch.Tensor,
        trajbyclass,
        layer,
        y_true: Optional[torch.Tensor] = None,
        getmatrix: bool = False,
        k: Optional[int] = None,
        t_k: float = 0.9,
        method: str = "ig",
        filter_onlycorrect: bool = False,
        op: str = "mean",
        norm: bool = False,
        seed: int = 0
    ):
        """
        Generate concept-based explanations for predictions.

        Args:
            x: Input trajectories
            trajbyclass: Trajectory examples by class
            layer: Layer to explain
            y_true: True labels (optional)
            getmatrix: Whether to return attribution matrix
            k: Number of top concepts (None for adaptive)
            t_k: Cumulative score threshold
            method: Attribution method ('ig', 'deeplift', 'nobackprop', 'random', 'identity')
            filter_onlycorrect: Only explain correct predictions
            op: Comparison operation ('mean', 'max', or None)
            norm: Whether to normalize attribution scores

        Returns:
            Top concept indices, attribution weights, and explanations
        """
        self.eval()
        s = time()
        # Get model predictions
        with torch.no_grad():
            x = x.to(self.device)
            class_scores, _, crelGs_raw, _ = self.forward(x, trainingmode=False)
        crelGs = self.output_activation(crelGs_raw) if norm else crelGs_raw
        y_pred = class_scores.argmax(dim=1).cpu()
        
        # Filter for correct predictions if requested
        if filter_onlycorrect and y_true is not None:
            x, y_pred, crelGs, class_scores = self._filter_correct_predictions(
                x, y_true, y_pred, crelGs, class_scores
            )
        # Compute attributions
        x_requires_grad = x.requires_grad_()
        targets = y_pred if y_true is None else y_pred

        attribution_weights = self._compute_attributions(
            x_requires_grad, targets, layer, method, seed
        ) # (batch, phis * classes)
        
        # Compute final attribution matrix
        final_matrix = self._compute_final_attributions(
            crelGs, attribution_weights, y_pred
        )
        # Get discriminative scores
        discriminative_scores = self._compute_discriminative_scores(
            final_matrix, targets, op
        )

        if getmatrix:
            grouped_matrix = self._group_matrix_by_class(final_matrix, targets)
            return grouped_matrix, y_pred
        
        # Generate explanations
        explanations = self._generate_explanations(
            x, y_true, y_pred, discriminative_scores, trajbyclass, k, t_k
        )
        return explanations

    def _filter_correct_predictions(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        crelGs: torch.Tensor,
        class_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter to keep only correctly predicted samples."""
        y_true = y_true.cpu()
        keep_indices = y_pred == y_true

        return (
            x[keep_indices],
            y_pred[keep_indices],
            crelGs[keep_indices],
            class_scores[keep_indices],
        )

    def _compute_attributions(
        self, x: torch.Tensor, targets: torch.Tensor, layer, method: str, seed:int=0
    ) -> torch.Tensor:
        """Compute attribution weights using specified method."""
        from contextlib import contextmanager

        @contextmanager
        def set_all_possible_seeds(s):
            torch.manual_seed(s)
            torch.cuda.manual_seed(s)
            torch.backends.cudnn.deterministic = True
            yield
    
        with set_all_possible_seeds(seed):
            if method == "nobackprop":
                return self._compute_nobackprop_attributions()
            elif method == "random":
                return self._compute_random_attributions()
            elif method == "identity":
                return self._compute_identity_attributions()
            elif method in ["deeplift", "ig"]:
                return self._compute_gradient_attributions(x, targets, layer, method)
            else:
                raise ValueError(f"Unknown attribution method: {method}")

    def _compute_nobackprop_attributions(self) -> torch.Tensor:
        """Compute output gradient attributions."""
        return self.classifier[-1].weight.cpu().detach()

    def _compute_random_attributions(self) -> torch.Tensor:
        """Compute random attributions for baseline."""
        return torch.randn_like(self.classifier[-1].weight.cpu().detach())

    def _compute_identity_attributions(self) -> torch.Tensor:
        """Compute identity attributions (all ones)."""
        return torch.ones_like(self.classifier[-1].weight.cpu().detach())

    def _compute_gradient_attributions(
        self, x: torch.Tensor, targets: torch.Tensor, layer, method: str
    ) -> torch.Tensor:
        """Compute gradient-based attributions (DeepLift or Integrated Gradients)."""

        wrapper_model = ForwardWrapper(self)

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        for p in layer.parameters():
            p.requires_grad_(True)
            
        if method == "deeplift":
            from captum.attr import LayerDeepLift
            attributor = LayerDeepLift(wrapper_model, layer)
        else:  # 'ig'
            from captum.attr import LayerIntegratedGradients
            attributor = LayerIntegratedGradients(wrapper_model, layer)

        baseline = torch.zeros_like(x[:1]).to(self.device)

        attribution_weights = (
            attributor.attribute(
                x,
                target=targets.to(x.device),
                baselines=baseline,
                attribute_to_layer_input=True,
                n_steps=15,
            )
            .detach()
        )
        
        for p in self.parameters():
            p.requires_grad_(True)

        return attribution_weights.cpu()

    def _compute_final_attributions(
        self,
        crelGs: torch.Tensor,
        attribution_weights: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute final attribution matrix."""
        if attribution_weights.dim() == 2:
            # For nobackprop, random, identity methods
            final_matrix = torch.zeros_like(crelGs).cpu()
            for i in range(len(crelGs)):
                final_matrix[i] = crelGs[i].cpu() * attribution_weights[y_pred[i]]
        else:
            # For gradient-based methods
            final_matrix = crelGs.cpu() * attribution_weights

        return final_matrix

    def _group_matrix_by_class(
        self, final_matrix: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Group attribution matrix by predicted class."""
        labels_expanded = targets.unsqueeze(1).expand(
            -1, final_matrix.size(1) // self.num_classes
        )
        offsets = torch.arange(
            0, final_matrix.size(1), self.num_classes, device=targets.device
        )
        indices = labels_expanded + offsets

        return torch.gather(final_matrix, 1, indices)

    def _compute_discriminative_scores(
        self, final_matrix: torch.Tensor, targets: torch.Tensor, op: str
    ) -> torch.Tensor:
        """Compute discriminative scores for concept selection."""
        grouped_matrix = self._group_matrix_by_class(final_matrix, targets)

        if op in ["mean", "max"]:
            batch_size = final_matrix.size(0)
            n_concepts = final_matrix.size(1) // self.num_classes
            concept_class_matrix = final_matrix.view(
                batch_size, n_concepts, self.num_classes
            )

            discriminative_scores = torch.zeros_like(grouped_matrix)

            for i in range(batch_size):
                target_class = targets[i]
                current_scores = concept_class_matrix[i]

                pred_scores = torch.abs(current_scores[:, target_class])

                other_classes = [
                    c for c in range(self.num_classes) if c != target_class
                ]
                other_scores = torch.abs(current_scores[:, other_classes])

                if op == "mean":
                    comparison_scores = other_scores.mean(dim=1)
                else:  # 'max'
                    comparison_scores = other_scores.max(dim=1).values

                discriminative_scores[i] = torch.abs(pred_scores - comparison_scores)
        else:
            discriminative_scores = torch.abs(grouped_matrix)

        return discriminative_scores

    def _generate_explanations(
        self,
        x: torch.Tensor,
        y_true: Optional[torch.Tensor],
        y_pred: torch.Tensor,
        discriminative_scores: torch.Tensor,
        trajbyclass,
        k: Optional[int],
        t_k: float,
    ) -> List:
        """Generate Explanation objects for each sample."""

        explanations = []

        if k is None:
            max_k = min(100, discriminative_scores.size(1)) # most will never use all concepts, no need to sort them all
            sorted_scores, sorted_indices = torch.topk(discriminative_scores, k = max_k, dim= 1)
            # sorted_scores, sorted_indices = torch.sort(
            #     discriminative_scores, descending=True, dim=1
            # )
            
            total_scores = discriminative_scores.sum(dim=1, keepdim=True)
            cumsum = torch.cumsum(sorted_scores, dim=1)
            cumsum_ratio = cumsum / total_scores.clamp(min=1e-8)
            
            # Find first index where cumsum >= t_k for each sample
            n_selected = (cumsum_ratio < t_k).sum(dim=1) + 1
            n_selected = n_selected.clamp(max=sorted_indices.size(1))
            
            # Adaptive top-k based on cumulative score threshold
            # sorted_indices = torch.argsort(
            #     discriminative_scores, descending=True, dim=1
            # )
            x_cpu = x.detach().cpu()
            y_pred_cpu = y_pred.cpu()
            y_true_cpu = y_true.cpu() if y_true is not None else None

            concepts = self.concepts
            for i in range(len(x)):
                n = n_selected[i].item()
                concept_indices = sorted_indices[i][:n].tolist()

            #     cumulative_score = 0
            #     total_score = discriminative_scores[i].sum().item()

            #     if total_score > 0:
            #         for n, idx in enumerate(sorted_indices[i]):
            #             cumulative_score += discriminative_scores[i][idx].item()
            #             if cumulative_score / total_score >= t_k:
            #                 break
            #         n_selected = n + 1
            #     else:
            #         n_selected = 0

            #     concept_indices = sorted_indices[i][:n_selected]
                top_concepts = [concepts[idx] for idx in concept_indices]

                explanations.append(
                    LocalExplanation(
                        trajectory=x_cpu[i],
                        true_label=y_true_cpu[i],
                        predicted_label=y_pred_cpu[i],
                        candidate_formulae=top_concepts,
                        trajectories_by_class=trajbyclass,
                    )
                )
        else:  #! not maintained
            # Fixed top-k
            top_indices = torch.topk(discriminative_scores, k=k, dim=1).indices

            for i in range(len(x)):
                top_concepts = [self.concepts[idx] for idx in top_indices[i]]

                explanations.append(
                    LocalExplanation(
                        trajectory=x[i].cpu(),
                        true_label=y_true[i].item() if y_true is not None else None,
                        predicted_label=y_pred[i].item(),
                        candidate_formulae=top_concepts,
                        trajectories_by_class=trajbyclass,
                    )
                )

        return explanations

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def set_robustness_info(self, robs_mean: torch.Tensor, robs_std: torch.Tensor):
        """Set robustness statistics for G_phis computation."""
        self.robs_mean = robs_mean
        self.robs_std = robs_std


@torch.jit.script
def compute_concept_relevance_jit(
    temb: torch.Tensor, crel_scale: torch.Tensor, crel_norm: bool
) -> torch.Tensor:
    """Compute concept_relevance weights with optional normalization."""
    scale = torch.exp(crel_scale).clamp(min=1e-6)
    if crel_norm:
        return torch.softmax(temb / scale, dim=1)
    return temb / scale


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, model) -> torch.Tensor:
    """
    Compute the total loss including classification and regularization terms.

    Args:
        logits: Model output logits
        labels: Target labels
        model: The model instance for accessing regularization parameters

    Returns:
        Total loss value
    """
    # Classification loss
    loss = F.cross_entropy(logits.float(), labels.max(1)[1], weight=model.weights)

    # concept_relevance collapse regularization
    if model.crel_scale is not None:
        collapse_strength = torch.exp(model.crel_collapse_strength)
        sigmoid_temp = torch.exp(model.log_sigmoid_temp)
        loss += collapse_strength * torch.sigmoid(-model.crel_scale / sigmoid_temp)

    # Epsilon control regularization
    if model.log_eps is not None:
        eps_strength = torch.exp(model.eps_control_strength)
        eps_term = torch.exp(model.log_eps) + torch.exp(-model.log_eps)
        loss += eps_strength * eps_term.squeeze()

    return loss