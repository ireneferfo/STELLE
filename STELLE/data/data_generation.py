"""
data_generation.py

Utilities to generate and plot synthetic multivariate time series trajectories
for classification experiments.

Functions:
- generate_synthetic_trajectories: create synthetic trajectories with controllable
    class separation, noise and drift.
- plot_trajectories: quick visualization helper to plot sample trajectories by class.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass


def generate_synthetic_trajectories(
    num_trajs: int,
    nvars: int,
    points: int,
    num_classes: int,
    seed: int = 0,
    # Classification difficulty controls
    class_separation: float = 1.0,  # Higher = easier to classify (0.5-2.0 recommended)
    inter_class_variance: float = 0.3,  # Variance within class (0.1-0.5 recommended)
    # Noise controls
    temporal_noise_std: float = 0.2,  # Gaussian noise on time series
    fourier_noise_std: float = 0.0,  # Noise added in frequency domain
    # Signal characteristics
    base_frequencies: Optional[list] = None,  # Base frequencies per class [f1, f2, ...]
    frequency_variation: float = 0.2,  # Random variation in frequencies
    periodicity_strength: float = 1.0,  # 0=weak periodic, 1=strong periodic
    # Additional controls
    drift_strength: float = 0.05,  # Linear drift magnitude
    outlier_prob: float = 0.0,  # Probability of outlier trajectories
    phase_coherence: float = 0.5,  # Phase relationship between variables (0-1)
    # Advanced options
    use_harmonics: bool = True,  # Add harmonic frequencies
    n_harmonics: int = 2,  # Number of harmonics to include
    trend_type: str = "linear",  # 'linear', 'quadratic', 'exponential', 'none'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate highly controllable synthetic multivariate trajectories.

    Parameters
    ----------
    num_trajs : int
        Number of trajectories (samples)
    nvars : int
        Number of variables (channels) per trajectory
    points : int
        Number of time points per trajectory
    num_classes : int
        Number of distinct classes
    seed : int
        RNG seed for reproducibility
    class_separation : float
        Controls how separated classes are. Higher = easier classification.
        Recommended: 0.5 (hard) to 2.0 (easy)
    inter_class_variance : float
        Within-class variance. Higher = more overlap = harder classification.
        Recommended: 0.1 (easy) to 0.5 (hard)
    temporal_noise_std : float
        Standard deviation of Gaussian noise added in time domain
    fourier_noise_std : float
        Standard deviation of noise added in frequency domain
    base_frequencies : list or None
        Base frequencies for each class. If None, auto-generated
    frequency_variation : float
        Random variation in frequencies between samples
    periodicity_strength : float
        0 = weak/irregular periodic patterns, 1 = strong periodic patterns
    drift_strength : float
        Magnitude of trend/drift component
    outlier_prob : float
        Probability of generating outlier trajectories (0-1)
    phase_coherence : float
        How synchronized phases are across variables (0=random, 1=synchronized)
    use_harmonics : bool
        Whether to include harmonic frequencies
    n_harmonics : int
        Number of harmonics to add
    trend_type : str
        Type of trend: 'linear', 'quadratic', 'exponential', 'none'

    Returns
    -------
    X : np.ndarray
        Trajectory data of shape (num_trajs, nvars, points)
    y : np.ndarray
        Class labels of shape (num_trajs,)
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize
    X = torch.zeros((num_trajs, nvars, points), dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_trajs,), dtype=torch.long)
    t = torch.linspace(0.0, 1.0, points)

    # Generate base frequencies for each class if not provided
    if base_frequencies is None:
        base_frequencies = [2.0 + i * 1.5 for i in range(num_classes)]

    # Generate class-specific parameters
    class_params = {}
    for cls in range(num_classes):
        class_params[cls] = {
            "base_freq": base_frequencies[cls % len(base_frequencies)],
            "amplitude": 1.0 + class_separation * cls * 0.3,
            "phase_offset": 2 * np.pi * cls / num_classes,
            "shape_param": cls / max(1, num_classes - 1),  # 0 to 1
        }

    for i in range(num_trajs):
        cls = int(y[i].item())
        is_outlier = np.random.rand() < outlier_prob

        # Get class-specific parameters with within-class variation
        params = class_params[cls]
        freq = params["base_freq"] * (1 + inter_class_variance * np.random.randn())
        amp = params["amplitude"] * (1 + inter_class_variance * np.random.randn())
        phase_base = params["phase_offset"]

        # Apply frequency variation
        freq *= 1 + frequency_variation * np.random.randn()

        for v in range(nvars):
            # Variable-specific frequency modulation
            var_freq = freq * (1 + 0.1 * v / max(1, nvars))

            # Phase: coherent across variables or random
            if phase_coherence > np.random.rand():
                phase = phase_base + 0.2 * v  # Coherent
            else:
                phase = 2 * np.pi * np.random.rand()  # Random

            # === Build base signal ===
            base = torch.zeros(points)

            # Primary periodic component
            primary = amp * torch.sin(2 * torch.pi * var_freq * t + phase)
            base += periodicity_strength * primary

            # Add harmonics
            if use_harmonics:
                for h in range(2, n_harmonics + 2):
                    harmonic_amp = amp / h
                    harmonic = harmonic_amp * torch.sin(
                        2 * torch.pi * var_freq * h * t + phase * h
                    )
                    base += periodicity_strength * harmonic * 0.5

            # Add non-periodic component (reduces pure periodicity)
            if periodicity_strength < 1.0:
                non_periodic_weight = 1.0 - periodicity_strength
                # Use different patterns based on class
                if params["shape_param"] < 0.33:
                    non_periodic = torch.exp(-((t - 0.5) ** 2) / 0.1)
                elif params["shape_param"] < 0.67:
                    non_periodic = torch.abs(torch.sin(5 * torch.pi * t))
                else:
                    non_periodic = t * (1 - t) * 4
                base += non_periodic_weight * amp * non_periodic

            # === Add trend/drift ===
            if trend_type == "linear":
                trend = drift_strength * v * (cls + 1) * t
            elif trend_type == "quadratic":
                trend = drift_strength * v * (cls + 1) * (t**2)
            elif trend_type == "exponential":
                trend = drift_strength * v * (cls + 1) * (torch.exp(t) - 1)
            else:  # 'none'
                trend = torch.zeros(points)

            base += trend

            # === Add temporal noise ===
            temporal_noise = temporal_noise_std * torch.randn(points)

            # === Add Fourier domain noise ===
            if fourier_noise_std > 0:
                # Transform to frequency domain
                fft_signal = torch.fft.rfft(base)

                # Add complex noise in frequency domain
                noise_real = fourier_noise_std * torch.randn(fft_signal.shape)
                noise_imag = fourier_noise_std * torch.randn(fft_signal.shape)
                fourier_noise = torch.complex(noise_real, noise_imag)

                fft_signal = fft_signal + fourier_noise

                # Transform back
                base = torch.fft.irfft(fft_signal, n=points)

            # === Handle outliers ===
            if is_outlier:
                # Make outlier significantly different
                base = base * (1 + 2 * np.random.randn()) + 3 * np.random.randn()

            # Combine all components
            X[i, v] = base + temporal_noise

    # Convert to numpy
    X = X.cpu().numpy()
    y = y.cpu().numpy()

    return X, y


@dataclass
class ExperimentConfig:
    """Dummy configuration."""

    n_train: int = 100
    n_test: int = 50
    nvars: int = 2
    series_length: int = 100
    num_classes: int = 3
    seed: int = 0



def get_synthetic_difficulty_params(difficulty: int) -> dict:
    """
    Get parameters for synthetic data generation based on difficulty level.

    Parameters
    ----------
    difficulty : int
        Difficulty level from 1 (easiest) to 10 (hardest)

    Returns
    -------
    dict
        Parameters to pass to generate_synthetic_trajectories
    """
    # Clamp difficulty to valid range
    
    # ANCHOR POINT: Difficulty 5 parameters (DO NOT CHANGE)
    anchor_params = {
        'class_separation': 1.5,
        'inter_class_variance': 0.3,
        'temporal_noise_std': 0.2,
        'fourier_noise_std': 0.125,
        'periodicity_strength': 0.7,
        'frequency_variation': 0.35,
        'drift_strength': 0.06,
        'outlier_prob': 0.04,
        'phase_coherence': 0.65,
        'use_harmonics': True,
        'n_harmonics': 2,
        'trend_type': 'linear',
    }
    
    # Return anchor if difficulty == 5
    if difficulty == 5:
        return anchor_params
    
    # EASY ENDPOINT: Difficulty 1 (recalibrated for clearer separation)
    easy_params = {
        'class_separation': 3.0,           # Much higher separation
        'inter_class_variance': 0.05,      # Very low variance
        'temporal_noise_std': 0.05,        # Minimal noise
        'fourier_noise_std': 0.0,          # No frequency noise
        'periodicity_strength': 1.0,       # Perfect periodicity
        'frequency_variation': 0.05,       # Very consistent frequencies
        'drift_strength': 0.01,            # Minimal drift
        'outlier_prob': 0.0,               # No outliers
        'phase_coherence': 1.0,            # Perfect coherence
        'use_harmonics': True,
        'n_harmonics': 3,
        'trend_type': 'linear',
    }
    
    # HARD ENDPOINT: Difficulty 10 (recalibrated for monotonic degradation)
    hard_params = {
        'class_separation': 0.4,           # Lower separation
        'inter_class_variance': 0.6,       # Higher variance
        'temporal_noise_std': 0.45,        # More noise
        'fourier_noise_std': 0.3,          # High frequency noise
        'periodicity_strength': 0.3,       # Weak periodicity
        'frequency_variation': 0.7,        # High frequency variation
        'drift_strength': 0.15,            # Strong drift
        'outlier_prob': 0.1,               # 10% outliers
        'phase_coherence': 0.2,            # Low coherence
        'use_harmonics': False,
        'n_harmonics': 1,
        'trend_type': 'exponential',
    }
    
    # Interpolate based on which side of anchor we're on
    if difficulty < 5:
        # Interpolate between easy (1) and anchor (5)
        t = (difficulty - 1) / 4.0  # 0 at diff=1, 1 at diff=5
        params = {}
        for key in anchor_params.keys():
            if key in ['use_harmonics', 'trend_type']:
                # Categorical parameters
                if key == 'use_harmonics':
                    params[key] = True
                elif key == 'trend_type':
                    params[key] = 'linear'
                elif key == 'n_harmonics':
                    params[key] = 3 if difficulty <= 2 else 2
            else:
                # Continuous parameters: linear interpolation
                params[key] = easy_params[key] + t * (anchor_params[key] - easy_params[key])
        
        # Handle n_harmonics
        if difficulty <= 2:
            params['n_harmonics'] = 3
        else:
            params['n_harmonics'] = 2
            
    else:  # difficulty > 5
        # Interpolate between anchor (5) and hard (10)
        t = (difficulty - 5) / 5.0  # 0 at diff=5, 1 at diff=10
        params = {}
        for key in anchor_params.keys():
            if key in ['use_harmonics', 'trend_type']:
                # Categorical parameters
                if key == 'use_harmonics':
                    params[key] = True if difficulty <= 7 else False
                elif key == 'trend_type':
                    if difficulty <= 7:
                        params[key] = 'linear'
                    elif difficulty <= 9:
                        params[key] = 'quadratic'
                    else:
                        params[key] = 'exponential'
                elif key == 'n_harmonics':
                    params[key] = 2 if difficulty <= 7 else 1
            else:
                # Continuous parameters: linear interpolation
                params[key] = anchor_params[key] + t * (hard_params[key] - anchor_params[key])
        
        # Handle categorical transitions
        if difficulty <= 7:
            params['n_harmonics'] = 2
            params['trend_type'] = 'linear'
        elif difficulty <= 9:
            params['n_harmonics'] = 1
            params['trend_type'] = 'quadratic'
        else:
            params['n_harmonics'] = 1
            params['trend_type'] = 'exponential'
    
    return params


    # difficulty = max(1, min(10, difficulty))

    # # Define parameter ranges
    # # Difficulty 1: Very easy (95%+ accuracy)
    # # Difficulty 5: Medium (85-90% accuracy)
    # # Difficulty 10: Very hard (60-70% accuracy)

    # # Linear interpolation between easy and hard parameters
    # t = (difficulty - 1) / 9.0  # Normalize to [0, 1]

    # params = {
    #     # Class separation: high for easy, low for hard
    #     "class_separation": 2.5 - t * 2.0,  # 2.5 -> 0.5
    #     # Within-class variance: low for easy, high for hard
    #     "inter_class_variance": 0.05 + t * 0.5,  # 0.05 -> 0.55
    #     # Temporal noise: low for easy, high for hard
    #     "temporal_noise_std": 0.05 + t * 0.35,  # 0.05 -> 0.40
    #     # Fourier noise: none for easy, high for hard
    #     "fourier_noise_std": t * 0.25,  # 0.0 -> 0.25
    #     # Periodicity: strong for easy, weak for hard
    #     "periodicity_strength": 1.0 - t * 0.6,  # 1.0 -> 0.4
    #     # Frequency variation: low for easy, high for hard
    #     "frequency_variation": 0.1 + t * 0.5,  # 0.1 -> 0.6
    #     # Drift: low for easy, higher for hard
    #     "drift_strength": 0.02 + t * 0.08,  # 0.02 -> 0.10
    #     # Outliers: none for easy, some for hard
    #     "outlier_prob": t * 0.08,  # 0.0 -> 0.08
    #     # Phase coherence: high for easy, low for hard
    #     "phase_coherence": 1.0 - t * 0.7,  # 1.0 -> 0.3
    #     # Harmonics: use for easier cases
    #     "use_harmonics": difficulty <= 7,
    #     "n_harmonics": max(1, 3 - difficulty // 3),  # 3 -> 1
    #     # Trend type: simpler for easy, more complex for hard
    #     "trend_type": (
    #         "linear"
    #         if difficulty <= 5
    #         else "quadratic" if difficulty <= 8 else "exponential"
    #     ),
    # }

    # return params


def load_data_with_difficulty(dataname: str, config: ExperimentConfig):
    """
    Load data based on dataset name, with support for synthetic difficulty levels.

    Parameters
    ----------
    dataname : str
        Dataset name. Use 'synthetic' or 'synthetic1' through 'synthetic10'
        where the number indicates difficulty (1=easiest, 10=hardest)
    config : ExperimentConfig
        Experiment configuration

    Returns
    -------
    X_train, y_train, X_test, y_test, num_classes
    """
    if "synthetic" in dataname:
        # Extract difficulty level from dataname
        if dataname == "synthetic":
            difficulty = 5  # Default medium difficulty
        else:
            try:
                # Extract number from 'synthetic3', 'synthetic10', etc.
                difficulty = int(dataname.replace("synthetic", ""))
                if difficulty < 1 or difficulty > 10:
                    raise ValueError(
                        f"Difficulty must be between 1 and 10, got {difficulty}"
                    )
            except ValueError:
                print(
                    f"Warning: Could not parse difficulty from '{dataname}', using default difficulty=5"
                )
                difficulty = 5

        # Get difficulty-specific parameters
        diff_params = get_synthetic_difficulty_params(difficulty)

        print(f"Generating synthetic data with difficulty={difficulty}")
        print(f"  Expected accuracy range: {get_expected_accuracy_range(difficulty)}")
        print(f"  Class separation: {diff_params['class_separation']:.2f}")
        print(f"  Inter-class variance: {diff_params['inter_class_variance']:.2f}")
        print(f"  Temporal noise: {diff_params['temporal_noise_std']:.2f}")

        # Generate training data
        X_train, y_train = generate_synthetic_trajectories(
            num_trajs=config.n_train,
            nvars=config.nvars,
            points=config.series_length,
            num_classes=config.num_classes,
            seed=config.seed,
            **diff_params,
        )

        # Generate test data
        X_test, y_test = generate_synthetic_trajectories(
            num_trajs=config.n_test,
            nvars=config.nvars,
            points=config.series_length,
            num_classes=config.num_classes,
            seed=config.seed + 1,
            **diff_params,
        )

        num_classes = config.num_classes

        return X_train, y_train, X_test, y_test, num_classes, diff_params

    else:
        # Handle other dataset types here
        raise NotImplementedError(f"Dataset '{dataname}' not implemented")


def get_expected_accuracy_range(difficulty: int) -> str:
    """Return expected accuracy range for a given difficulty level."""
    ranges = {
        1: "95-99%",
        2: "92-96%",
        3: "88-93%",
        4: "85-90%",
        5: "82-88%",
        6: "78-85%",
        7: "74-82%",
        8: "70-78%",
        9: "65-73%",
        10: "60-70%",
    }
    return ranges.get(difficulty, "Unknown")


# Example usage
if __name__ == "__main__":
    config = ExperimentConfig(
        n_train=100, n_test=50, nvars=3, series_length=50, num_classes=3, seed=42
    )

    # Test different difficulty levels
    for dataname in ["synthetic1", "synthetic5", "synthetic10"]:
        print(f"\n{'='*60}")
        print(f"Testing {dataname}")
        print("=" * 60)

        X_train, y_train, X_test, y_test, num_classes = load_data_with_difficulty(
            dataname, config
        )

        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"Class distribution: {np.bincount(y_train)}")


# Example usage showing difficulty control
def demonstrate_difficulty_levels():
    """Show how to generate datasets with different difficulty levels"""

    # Easy classification (90-95% accuracy expected)
    X_easy, y_easy = generate_synthetic_trajectories(
        num_trajs=1000,
        nvars=5,
        points=100,
        num_classes=3,
        class_separation=2.0,  # High separation
        inter_class_variance=0.1,  # Low within-class variance
        temporal_noise_std=0.1,  # Low noise
        fourier_noise_std=0.0,  # No frequency noise
        periodicity_strength=1.0,  # Strong periodic patterns
    )

    # Medium difficulty (80-90% accuracy expected)
    X_medium, y_medium = generate_synthetic_trajectories(
        num_trajs=1000,
        nvars=5,
        points=100,
        num_classes=3,
        class_separation=1.0,  # Moderate separation
        inter_class_variance=0.3,  # Moderate variance
        temporal_noise_std=0.2,  # Moderate noise
        fourier_noise_std=0.1,  # Some frequency noise
        periodicity_strength=0.7,  # Mixed periodic/non-periodic
        frequency_variation=0.3,
    )

    # Hard classification (60-75% accuracy expected)
    X_hard, y_hard = generate_synthetic_trajectories(
        num_trajs=1000,
        nvars=5,
        points=100,
        num_classes=3,
        class_separation=0.5,  # Low separation
        inter_class_variance=0.5,  # High within-class variance
        temporal_noise_std=0.3,  # High noise
        fourier_noise_std=0.2,  # High frequency noise
        periodicity_strength=0.5,  # Weak periodicity
        outlier_prob=0.05,  # 5% outliers
        frequency_variation=0.5,
    )

    return X_easy, y_easy, X_medium, y_medium, X_hard, y_hard


def plot_trajectories(
    X: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 3,
    samples_per_class: int = 5,
    path=None,
) -> None:
    """
    Plot sample trajectories for each variable and class.

    Parameters
    - X: tensor (num_trajs, nvars, points)
    - y: tensor (num_trajs,)
    - num_classes: number of classes to show (colors indexed by class id)
    - samples_per_class: how many examples per class to plot
    """
    _, nvars, _ = X.shape

    # Configure figure size: one subplot per variable
    plt.figure(figsize=(15, max(2, nvars * 2)))
    colors = plt.cm.tab10.colors  # color palette for up to 10 classes

    for var in range(nvars):
        plt.subplot(nvars, 1, var + 1)
        for cls in range(num_classes):
            # Indices of samples belonging to this class
            class_indices = (y == cls).nonzero()[0][:samples_per_class]
            for idx in class_indices:
                # Plot trajectory for this sample and variable
                plt.plot(
                    X[int(idx), var],
                    label=f"class {cls}" if var == 0 else "",
                    color=colors[cls % len(colors)],
                )
        plt.title(f"Variable {var}")
        plt.xticks([])

    # Place a single legend (only uses labels from first subplot)
    plt.legend()
    plt.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
