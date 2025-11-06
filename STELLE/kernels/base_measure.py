"""
BaseMeasure - Base classes for sampling trajectories and generating STL formulae.
"""

import torch

class Measure:
    """Abstract base class for trajectory sampling measures."""
    
    def sample(self, samples: int = 100000, varn: int = 2, points: int = 100) -> torch.Tensor:
        """
        Sample trajectories from the measure.
        
        Args:
            samples: Number of trajectories to sample
            varn: Number of variables per trajectory
            points: Number of points per trajectory
            
        Returns:
            Tensor of shape (samples, varn, points) containing sampled trajectories
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement sample method")


class BaseMeasure(Measure):
    """
    Basic measure for sampling trajectories with normal distributions.
    
    Samples trajectories where:
    - Initial state follows normal distribution
    - Total variation follows normal distribution  
    - Trajectories are piecewise linear with random derivative changes
    
    Most trajectories will fall within [-4, 4] range due to standard normal parameters.
    """
    
    def __init__(
        self,
        initial_mean: float = 0.0, # mu0
        initial_std: float = 1.0, # sigma0
        variation_mean: float = 0.0, # mu1
        variation_std: float = 1.0, # sigma1
        sign_change_prob: float = 0.1, # q
        initial_sign_prob: float = 0.5, # q0
        device: str = "cpu",
        density: int = 1,
    ):
        """
        Initialize the base measure.
        
        Args:
            initial_mean: Mean of normal distribution for initial state
            initial_std: Standard deviation of normal distribution for initial state
            variation_mean: Mean of normal distribution for total variation
            variation_std: Standard deviation of normal distribution for total variation
            sign_change_prob: Probability of derivative sign change at each point
            initial_sign_prob: Probability of initial positive derivative sign
            device: Device to run computations on ('cpu' or 'cuda' or 'mps')
            density: Density factor for trajectory interpolation (1 = no interpolation)
        """
        self.initial_mean = initial_mean
        self.initial_std = initial_std
        self.variation_mean = variation_mean
        self.variation_std = variation_std
        self.sign_change_prob = sign_change_prob
        self.initial_sign_prob = initial_sign_prob
        self.device = device
        self.density = density
        
        self._validate_parameters()
        self._setup_device()
        self._set_random_seeds()

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        if self.sign_change_prob < 0 or self.sign_change_prob > 1:
            raise ValueError("sign_change_prob must be between 0 and 1")
        if self.initial_sign_prob < 0 or self.initial_sign_prob > 1:
            raise ValueError("initial_sign_prob must be between 0 and 1")
        if self.density < 1:
            raise ValueError("density must be at least 1")

    def _setup_device(self) -> None:
        """Setup computation device and validate CUDA availability."""
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU card or CUDA library not available")

    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(0)
        if self.device == "cuda":
            torch.cuda.manual_seed(0)

    def sample(
        self, 
        samples: int = 10000, 
        varn: int = 1, 
        points: int = 100
    ) -> torch.Tensor:
        """
        Sample trajectories from the base measure space.
        
        Args:
            samples: Number of trajectories to sample
            varn: Number of variables per trajectory
            points: Number of points per trajectory
            
        Returns:
            Tensor of shape (samples, varn, points) containing sampled trajectories
        """
        # Generate uniform random numbers for trajectory construction
        trajectories = self._generate_base_trajectories(samples, varn, points)
        
        # Apply derivative patterns
        derivatives = self._generate_derivative_patterns(samples, varn, points)
        
        # Apply total variation scaling
        total_variation = self._sample_total_variation(samples, varn)
        derivatives = derivatives * total_variation
        derivatives[:, :, 0] = 1.0  # Make initial point non-invasive
        
        # Construct final trajectories
        trajectories = trajectories * derivatives
        trajectories = torch.cumsum(trajectories, 2)
        
        # Apply density interpolation if needed
        if self.density > 1:
            trajectories = self._apply_density_interpolation(trajectories, points)
            
        return trajectories

    def _generate_base_trajectories(
        self, 
        samples: int, 
        varn: int, 
        points: int
    ) -> torch.Tensor:
        """Generate base trajectories using uniform random numbers and sorting."""
        trajectories = torch.rand(samples, varn, points, device=self.device)
        
        # Set first point to 0 and last point to 1 for normalization
        trajectories[:, :, 0] = 0.0
        trajectories[:, :, -1] = 1.0
        
        # Sort each trajectory to create monotonic sequences
        trajectories, _ = torch.sort(trajectories, 2)
        
        # Convert to increments
        trajectories[:, :, 1:] = trajectories[:, :, 1:] - trajectories[:, :, :-1]
        
        # Set initial state from normal distribution
        trajectories[:, :, 0] = (
            self.initial_mean + 
            self.initial_std * torch.randn(trajectories[:, :, 0].size(), device=self.device)
        )
        
        return trajectories

    def _generate_derivative_patterns(
        self, 
        samples: int, 
        varn: int, 
        points: int
    ) -> torch.Tensor:
        """Generate derivative sign patterns using Bernoulli distributions."""
        # Sample derivative sign changes
        derivative_signs = (1 - self.sign_change_prob) * torch.ones(
            samples, varn, points, device=self.device
        )
        derivative_signs = 2 * torch.bernoulli(derivative_signs) - 1
        
        # Sample initial derivative sign
        derivative_signs[:, :, 0] = self.initial_sign_prob
        derivative_signs[:, :, 0] = 2 * torch.bernoulli(derivative_signs[:, :, 0]) - 1
        
        # Cumulative product to get persistent sign changes
        return torch.cumprod(derivative_signs, 2)

    def _sample_total_variation(self, samples: int, varn: int) -> torch.Tensor:
        """Sample total variation from normal distribution."""
        return torch.pow(
            self.variation_mean + 
            self.variation_std * torch.randn(samples, varn, 1, device=self.device),
            2,
        )

    def _apply_density_interpolation(
        self, 
        trajectories: torch.Tensor, 
        original_points: int
    ) -> torch.Tensor:
        """Apply linear interpolation to increase trajectory point density."""
        samples, varn, _ = trajectories.shape
        new_points = (original_points - 1) * self.density + 1
        
        dense_trajectories = torch.zeros(samples, varn, new_points, device=self.device)
        dense_trajectories[:, :, ::self.density] = trajectories
        
        # Compute differences between original points
        differences = (
            dense_trajectories[:, :, self.density::self.density] - 
            dense_trajectories[:, :, 0:-self.density:self.density]
        )
        
        # Fill in interpolated points
        for i in range(self.density - 1):
            dense_trajectories[:, :, i + 1::self.density] = (
                dense_trajectories[:, :, 0:-self.density:self.density] + 
                (differences / self.density) * (i + 1)
            )
            
        return dense_trajectories