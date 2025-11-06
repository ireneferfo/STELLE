"""
A fully-differentiable implementation of Signal Temporal Logic semantic trees.

This module provides a PyTorch-based implementation of Signal Temporal Logic (STL)
with both boolean and quantitative (robustness) semantics. The implementation
supports various STL operators including atomic propositions, logical operators
(AND, OR, NOT), and temporal operators (ALWAYS, EVENTUALLY, UNTIL).

Key Features:
- Fully differentiable robustness semantics
- Support for both bounded and unbounded temporal operators
- GPU acceleration support via PyTorch
- Memory-efficient implementation
"""

# For custom type-hints
from typing import Union

realnum = Union[float, int]

# For tensor functions
import torch
from torch import Tensor
import torch.nn.functional as F
import os

# TODO: automatic check of timespan when evaluating robustness? (should be done only at root node)

# Environment variable for MPS (Metal Performance Shaders) fallback on macOS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'


def eventually(x: Tensor, time_span: int) -> Tensor:
    """
    STL operator 'eventually' in 1D using max pooling.
    
    Computes the maximum value over a sliding window of size time_span,
    which corresponds to the eventually operator in STL semantics.
    
    Note: Currently requires time_span to be an integer (working with discrete steps).
    
    Args:
        x: Input signal tensor of shape (batch_size, channels, time_steps)
        time_span: Size of the sliding window (must be integer)
        
    Returns:
        Tensor: Result of applying eventually operator along time dimension
    """
    # mps doesnt support .double()
    # if torch.backends.mps.is_available(): x= x.float()
    # else: x = x.double()
    # Convert boolean tensors to float for pooling operations
    if x.dtype == torch.bool:
        x = x.float()
    # Use max pooling to compute maximum over sliding window
    return F.max_pool1d(x, kernel_size=time_span, stride=1)


class Node:
    """
    Abstract base class for all STL formula nodes.
    
    This class defines the interface for all STL nodes and provides
    common functionality for evaluating both boolean and quantitative semantics.
    """

    def __init__(self) -> None:
        """Initialize the node. Must be implemented by subclasses."""
        pass

    def __str__(self) -> str:
        """String representation of the node. Must be implemented by subclasses."""
        pass

    def boolean(self, x: Tensor, evaluate_at_all_times: bool = False) -> Tensor:
        """
        Evaluate the boolean semantics of this STL formula.
        
        Args:
            x: Input signals tensor of shape (N_samples, N_vars, N_sampling_points)
            evaluate_at_all_times: If True, returns semantics for all time points;
                                 if False, returns only at time t=0
                                 
        Returns:
            Boolean semantics tensor
        """
        z: Tensor = self._boolean(x)
        if evaluate_at_all_times:
            return z
        else:
            return self._extract_semantics_at_time_zero(z)

    def quantitative(
        self,
        x: Tensor,
        evaluate_at_all_times: bool = False,
        vectorize: bool = False,
        normalize: bool = False,
    ) -> Tensor:
        """
        Evaluate the quantitative (robustness) semantics of this STL formula.
        
        Args:
            x: Input signals tensor of shape (N_samples, N_vars, N_sampling_points)
            evaluate_at_all_times: If True, returns semantics for all time points;
                                 if False, returns only at time t=0
            vectorize: If True, returns robustness for all variables;
                      if False, returns aggregated robustness
            normalize: If True, normalizes robustness values (experimental)
                      
        Returns:
            Quantitative robustness semantics tensor
        """
        # Ensure appropriate precision for different hardware
        if torch.backends.mps.is_available(): 
            x = x.float()
        else: 
            x = x.double()
            
        z: Tensor = self._quantitative(x, vectorize, normalize)
        if evaluate_at_all_times:
            return z
        else:
            return self._extract_semantics_at_time_zero(z)

    def set_normalizing_flag(self, value: bool = True) -> None:
        """
        Set normalization flag for robustness values.
        
        Currently not fully implemented/used.
        """
        pass

    def time_depth(self) -> int:
        """
        Calculate the time depth of bounded temporal operators.
        
        Returns:
            Maximum time depth needed to evaluate this formula
        """
        pass

    def _quantitative(
        self, x: Tensor, vectorize: bool = False, normalize: bool = False
    ) -> Tensor:
        """Internal method for quantitative semantics. Implemented by subclasses."""
        pass

    def _boolean(self, x: Tensor) -> Tensor:
        """Internal method for boolean semantics. Implemented by subclasses."""
        pass

    @staticmethod
    def _extract_semantics_at_time_zero(x: Tensor) -> Tensor:
        """
        Extract semantics values at time zero.
        
        Args:
            x: Semantics tensor of shape (batch_size, channels, time_steps)
            
        Returns:
            Tensor of shape (batch_size, channels) with values at time zero
        """
        return x[:, :, 0]
    

class Boolean(Node):
    """
    Node representing a constant boolean value (True or False).
    
    This is useful for creating constant STL formulae.
    """

    def __init__(self, value: bool):
        """
        Initialize boolean node.
        
        Args:
            value: The boolean value (True or False)
        """
        super().__init__()
        self.value = value

    def __eq__(self, other):
        """Check equality with another boolean node."""
        return isinstance(other, Boolean) and self.value == other.value

    def __str__(self):
        """String representation."""
        return "True" if self.value else "False"

    def time_depth(self) -> int:
        """Boolean constants have zero time depth."""
        return 0

    def _boolean(self, x: Tensor) -> Tensor:
        """Return constant boolean value replicated to match input shape."""
        return self.value

    def _quantitative(
        self, x: Tensor, vectorize: bool = False, normalize: bool = False
    ) -> Tensor:
        """
        Return quantitative semantics for boolean constant.
        
        For True: returns +infinity (maximally satisfied)
        For False: returns -infinity (maximally violated)
        """
        # Create tensor with appropriate shape
        shape = (x.shape[0], 1, x.shape[2]) if not vectorize else x.shape
        
        # Use infinity values to represent maximal satisfaction/violation
        robustness = torch.full(
            shape, float("inf") if self.value else float("-inf"), device=x.device
        )

        # Experimental normalization (currently commented out)
        if normalize:
            robustness = torch.tanh(robustness)

        return robustness


class Atom(Node):
    """
    Atomic formula node representing variable comparisons.
    
    Represents formulae of the form: x_i <= threshold or x_i >= threshold
    where x_i is a signal variable and threshold is a real number.
    """

    def __init__(self, var_index: int, threshold: realnum, lte: bool = False) -> None:
        """
        Initialize atomic formula.
        
        Args:
            var_index: Index of the variable to compare
            threshold: Comparison threshold value
            lte: If True, uses <= comparison; if False, uses >= comparison
        """
        super().__init__()
        self.var_index: int = var_index
        self.threshold: realnum = threshold
        self.lte: bool = lte  # Less Than or Equal flag

    def __str__(self) -> str:
        """String representation of the atomic formula."""
        s: str = (
            "x_"
            + str(self.var_index)
            + (" <= " if self.lte else " >= ")
            + str(round(float(self.threshold), 4))
        )
        return s

    def time_depth(self) -> int:
        """Atomic formulae have zero time depth."""
        return 0

    def _boolean(self, x: Tensor) -> Tensor:
        """
        Evaluate boolean semantics for atomic formula.
        
        Args:
            x: Input signal tensor
            
        Returns:
            Boolean tensor indicating where the condition holds
        """
        # Extract the specific variable we're interested in
        xj: Tensor = x[:, self.var_index, :]
        xj: Tensor = xj.view(xj.size()[0], 1, -1)  # Reshape for consistency
        
        # Apply the comparison operator
        z: Tensor = (
            torch.le(xj, self.threshold) if self.lte else torch.ge(xj, self.threshold)
        )
        return z

    def _quantitative(
        self, x: Tensor, vectorize: bool = False, normalize: bool = False
    ) -> Tensor:
        """
        Evaluate quantitative semantics for atomic formula.
        
        The robustness is defined as:
          For x <= c: robustness = c - x
          For x >= c: robustness = x - c
          
        Positive values indicate satisfaction, negative values indicate violation.
        """
        # Extract the specific variable
        xj: Tensor = x[:, self.var_index, :]

        if not vectorize:
            xj: Tensor = xj.view(xj.size()[0], 1, -1)

        # Compute robustness based on comparison type
        if self.lte:
            zj: Tensor = -xj + self.threshold  # c - x for x <= c
        else:
            zj: Tensor = xj - self.threshold   # x - c for x >= c
        del xj  # Free memory

        # Experimental normalization (currently commented out)
        if normalize:
            zj: Tensor = torch.tanh(zj)

        # For vectorized output, return robustness for all variables
        if vectorize:
            z = torch.zeros_like(x)
            z[:, self.var_index, :] = zj
            return z
        return zj


class Not(Node):
    """Negation node."""

    def __init__(self, child: Node) -> None:
        super().__init__()
        self.child: Node = child

    def __str__(self) -> str:
        s: str = "not ( " + self.child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        return self.child.time_depth()

    def _boolean(self, x: Tensor) -> Tensor:
        z: Tensor = ~self.child._boolean(x)
        return z

    def _quantitative(
        self, x: Tensor, vectorize: bool = False, normalize: bool = False
    ) -> Tensor:
        z: Tensor = -self.child._quantitative(x, vectorize, normalize)
        return z


class And(Node):
    """Conjunction node."""

    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child

    def __str__(self) -> str:
        s: str = (
            "( "
            + self.left_child.__str__()
            + " and "
            + self.right_child.__str__()
            + " )"
        )
        return s

    def time_depth(self) -> int:
        return max(self.left_child.time_depth(), self.right_child.time_depth())

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.left_child._boolean(x).cpu()
        z2: Tensor = self.right_child._boolean(x).cpu()
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.logical_and(z1, z2)
        return z.cpu()

    def _quantitative(
        self, x: Tensor, vectorize: bool = False, normalize: bool = False
    ) -> Tensor:
        """
        (0,0) -> 0
        (a,0) -> a
        (a,b) -> min(a,b)
        """
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        z1: Tensor = self.left_child._quantitative(x, vectorize, normalize).to(dev)
        z2: Tensor = self.right_child._quantitative(x, vectorize, normalize).to(dev)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size].to(dev)
        z2: Tensor = z2[:, :, :size].to(dev)

        if vectorize:
            z: Tensor = torch.minimum(z1, z2) + torch.maximum(z1, z2) * (
                torch.minimum(z1, z2) == 0
            )
        else:
            z: Tensor = torch.min(z1, z2)
        del z1, z2
        return z.cpu()


class Or(Node):
    """Disjunction node."""

    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child

    def __str__(self) -> str:
        s: str = (
            "( "
            + self.left_child.__str__()
            + " or "
            + self.right_child.__str__()
            + " )"
        )
        return s

    def time_depth(self) -> int:
        return max(self.left_child.time_depth(), self.right_child.time_depth())

    def _boolean(self, x: Tensor) -> Tensor:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z1: Tensor = self.left_child._boolean(x).to(dev)
        z2: Tensor = self.right_child._boolean(x).to(dev)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.logical_or(z1, z2)
        return z.cpu()

    def _quantitative(
        self, x: Tensor, vectorize: bool = False, normalize: bool = False
    ) -> Tensor:
        """
        (0,0) -> 0
        (a,0) -> a
        (a,b) -> max(a,b)
        """
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        z1: Tensor = self.left_child._quantitative(x, vectorize, normalize).to(dev)
        z2: Tensor = self.right_child._quantitative(x, vectorize, normalize).to(dev)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = z1 + z2 - torch.minimum(z1, z2) if vectorize else torch.max(z1, z2)
        del z1, z2
        return z.cpu()


class Globally(Node):
    """Globally node."""

    # # if Globally[0.0] is being instantiated, behave as its child
    # def __new__(cls, *args, **kwargs):
    #     # Extract relevant args for early return check
    #     if len(args) >= 1:
    #         child = args[0]
    #         left_time_bound = kwargs.get("left_time_bound", 0)
    #         right_time_bound = kwargs.get("right_time_bound", 1)
    #     else:
    #         # Handle case where child is passed as a keyword argument
    #         child = kwargs.get("child")
    #         left_time_bound = kwargs.get("left_time_bound", 0)
    #         right_time_bound = kwargs.get("right_time_bound", 1)

    #     if left_time_bound == 0 and right_time_bound == 0:
    #         return child  # Return the child directly
    #     return super().__new__(cls)  # Create instance of Globally

    def __init__(
        self,
        child: Node,
        unbound: bool = False,
        right_unbound: bool = False,
        left_time_bound: int = 0,
        right_time_bound: int = 1,
        adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound  #  + 1
        self.adapt_unbound: bool = adapt_unbound
        if right_unbound is True and left_time_bound == 0:
            self.unbound = True

        if (
            (self.unbound is False)
            and (self.right_unbound is False)
            and (self.right_time_bound <= self.left_time_bound)
        ):
            raise ValueError(
                f"Temporal thresholds are incorrect: right parameter ({self.right_time_bound}) <= left parameter ({self.left_time_bound})"
            )

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "always" + s0 + " ( " + self.child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth()
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return self.child.time_depth() + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(
            x[:, :, self.left_time_bound :]
        )  # nested temporal parameters
        # z1 = z1[:, :, self.left_time_bound:]
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummin(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.min(z1, 2, keepdim=True)
        else:
            try:
                z: Tensor = torch.ge(
                    1.0
                    - eventually(
                        (~z1), self.right_time_bound - self.left_time_bound
                    ),
                    0.5,)
            except: print('boolean eventually problem in: ', self)
        del z1
        return z.cpu()

    def _quantitative(
        self, x: Tensor, vectorize: bool = False, normalize: bool = False
    ) -> Tensor:
        torch.cuda.empty_cache()
        z1: Tensor = self.child._quantitative(
            x[:, :, self.left_time_bound :], vectorize, normalize
        )
        # z1 = z1[:, :, self.left_time_bound:]
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummin(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.min(z1, 2, keepdim=True)
        else:
            try: z: Tensor = -eventually(-z1, self.right_time_bound - self.left_time_bound)
            except: print('quantitative eventually problem in: ', self)

        del z1
        return z.cpu()


class Eventually(Node):
    """Eventually node."""

    def __init__(
        self,
        child: Node,
        unbound: bool = False,
        right_unbound: bool = False,
        left_time_bound: int = 0,
        right_time_bound: int = 1,
        adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound  # + 1
        self.adapt_unbound: bool = adapt_unbound
        if right_unbound is True and left_time_bound == 0:
            self.unbound = True

        if (
            (self.unbound is False)
            and (self.right_unbound is False)
            and (self.right_time_bound <= self.left_time_bound)
        ):
            raise ValueError(
                f"Temporal thresholds are incorrect: right parameter ({self.right_time_bound}) <= left parameter ({self.left_time_bound})"
            )

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "eventually" + s0 + " ( " + self.child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth()
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return self.child.time_depth() + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(x[:, :, self.left_time_bound :])
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 2, keepdim=True)
        else:
            # try: z: Tensor = torch.ge(
            #     eventually(z1, self.right_time_bound - self.left_time_bound),
            #     0.5,
            # )
            try:
                z = eventually(z1, self.right_time_bound - self.left_time_bound)
                z = z >= 0.5   # back to bool
            except Exception as e: 
                print('\nboolean eventually problem in: ', self)
                print(f'{x.shape=}, {z1.shape=}')
                print(e)
        del z1
        return z.cpu()

    def _quantitative(
        self, x: Tensor, vectorize: bool = False, normalize: bool = False
    ) -> Tensor:
        torch.cuda.empty_cache()
        z1: Tensor = self.child._quantitative(
            x[:, :, self.left_time_bound :], vectorize, normalize
        )
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 2, keepdim=True)
        else:
            try: z: Tensor = eventually(z1, self.right_time_bound - self.left_time_bound)
            except Exception as e: 
                print('\nquantitative eventually problem in: ', self)
                print(f'{x.shape=}, {z1.shape=}')
                print(e)

        del z1
        return z.cpu()


class Until(Node):
    # TODO: maybe define timed and untimed until, and use this class to wrap them
    """Until node."""

    def __init__(
        self,
        left_child: Node,
        right_child: Node,
        unbound: bool = False,
        right_unbound: bool = False,
        left_time_bound: int = 0,
        right_time_bound: int = 1,
    ) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound  # + 1
        if right_unbound is True and left_time_bound == 0:
            self.unbound = True

        if (
            (self.unbound is False)
            and (self.right_unbound is False)
            and (self.right_time_bound <= self.left_time_bound)
        ):
            raise ValueError(
                f"Temporal thresholds are incorrect in {self}: right parameter {self.right_time_bound} is higher than left parameter {self.left_time_bound}"
            )

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = (
            "( "
            + self.left_child.__str__()
            + " until"
            + s0
            + " "
            + self.right_child.__str__()
            + " )"
        )
        return s

    def time_depth(self) -> int:
        sum_children_depth: int = (
            self.left_child.time_depth() + self.right_child.time_depth()
        )
        if self.unbound:
            return sum_children_depth
        elif self.right_unbound:
            return sum_children_depth + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return sum_children_depth + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.unbound:
            z1: Tensor = self.left_child._boolean(x).to(dev)
            z2: Tensor = self.right_child._boolean(x).to(dev)
            size: int = min(z1.size()[2], z2.size()[2])
            z1: Tensor = z1[:, :, :size]
            z2: Tensor = z2[:, :, :size]
            z1_rep = torch.repeat_interleave(
                z1.unsqueeze(2), z1.unsqueeze(2).shape[-1], 2
            )
            z1_tril = torch.tril(z1_rep.transpose(2, 3), diagonal=-1)
            z1_triu = torch.triu(z1_rep)
            z1_def = torch.cummin(z1_tril + z1_triu, dim=3)[0]

            z2_rep = torch.repeat_interleave(
                z2.unsqueeze(2), z2.unsqueeze(2).shape[-1], 2
            )
            z2_tril = torch.tril(z2_rep.transpose(2, 3), diagonal=-1)
            z2_triu = torch.triu(z2_rep)
            z2_def = z2_tril + z2_triu
            z: Tensor = torch.max(
                torch.min(
                    torch.cat([z1_def.unsqueeze(-1), z2_def.unsqueeze(-1)], dim=-1),
                    dim=-1,
                )[0],
                dim=-1,
            )[0]
        elif self.right_unbound:
            timed_until: Node = And(
                Globally(
                    self.left_child,
                    left_time_bound=0,
                    right_time_bound=self.left_time_bound,
                ),
                And(
                    Eventually(
                        self.right_child,
                        right_unbound=True,
                        left_time_bound=self.left_time_bound,
                    ),
                    Eventually(
                        Until(self.left_child, self.right_child, unbound=True),
                        left_time_bound=self.left_time_bound,
                        right_unbound=True,
                    ),
                ),
            )
            z: Tensor = timed_until._boolean(x)
        else:
            if self.left_time_bound == 0:
                # φ₁ U[0,b] φ₂ ≡ ◇[0,b] φ₂ ∧ (□[0,b] (φ₁ ∨ φ₂))
                timed_until = And(
                    Eventually(
                        self.right_child,
                        left_time_bound=0,
                        right_time_bound=self.right_time_bound,
                    ),
                    Globally(
                        Or(self.left_child, self.right_child),
                        left_time_bound=0,
                        right_time_bound=self.right_time_bound,
                    ),
                )
            else:
                timed_until: Node = And(
                    Globally(
                        self.left_child,
                        left_time_bound=0,
                        right_time_bound=self.left_time_bound,
                    ),
                    And(
                        Eventually(
                            self.right_child,
                            left_time_bound=self.left_time_bound,
                            right_time_bound=self.right_time_bound,  # - 1,
                        ),
                        Eventually(
                            Until(self.left_child, self.right_child, unbound=True),
                            left_time_bound=self.left_time_bound,
                            right_unbound=True,
                        ),
                    ),
                )
            z: Tensor = timed_until._boolean(x)
        return z.cpu()

    def _quantitative(
        self, x: Tensor, vectorize: bool = False, normalize: bool = False
    ) -> Tensor:
        torch.cuda.empty_cache()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.unbound:
            """
            (0,0) -> 0
            (a,0) -> a
            (a,b) -> max(cummin(a),b)
            """
            with torch.no_grad():
                z1: Tensor = self.left_child._quantitative(x, vectorize, normalize).to(
                    dev
                )
                z2: Tensor = self.right_child._quantitative(x, vectorize, normalize).to(
                    dev
                )
                size: int = min(z1.size()[2], z2.size()[2])
                z1: Tensor = z1[:, :, :size]
                z2: Tensor = z2[:, :, :size]
                # print(f'{z1=}\n{z2=}')
                # Process in chunks to reduce memory usage
                z_list = []
                for t in range(size):
                    z1_t = z1[:, :, t:].unsqueeze(-1)
                    z2_t = z2[:, :, t:].unsqueeze(-1)
                    if vectorize:
                        z1_cummin = torch.cummin(z1_t, dim=2)[0]
                        zero_mask = (z1_cummin == 0) & (z2_t == 0)
                        z_min = torch.minimum(z1_cummin, z2_t)
                        z_min = torch.where(zero_mask, torch.zeros_like(z_min), z_min)
                    else:
                        z_min = torch.min(
                            torch.cat([torch.cummin(z1_t, dim=2)[0], z2_t], dim=-1),
                            dim=-1,
                        )[0]
                        # if t==0: print(f'{z_min=}')
                    z_max = torch.max(z_min, dim=2, keepdim=True)[0]
                    # if t==0: print(f'{z_max=}')
                    z_list.append(z_max)
                    del z1_t, z2_t, z_min, z_max
                    torch.cuda.empty_cache()

                z = (
                    torch.cat(z_list, dim=2).squeeze(-1)
                    if vectorize
                    else torch.cat(z_list, dim=2)
                )
                del z1, z2, z_list

                # z: Tensor = torch.cat([torch.max(torch.min(
                #    torch.cat([torch.cummin(z1[:, :, t:].unsqueeze(-1), dim=2)[0], z2[:, :, t:].unsqueeze(-1)], dim=-1),
                #    dim=-1)[0], dim=2, keepdim=True)[0] for t in range(size)], dim=2)
                # del z1, z2

        elif self.right_unbound:
            timed_until: Node = And(
                    Globally(
                        self.left_child,
                        left_time_bound=0,
                        right_time_bound=self.left_time_bound,
                    ),
                    And(
                        Eventually(
                            self.right_child,
                            left_time_bound=self.left_time_bound,
                            right_unbound=True,
                        ),
                        Eventually(
                            Until(self.left_child, self.right_child, unbound=True),
                            left_time_bound=self.left_time_bound,
                            right_unbound=True,
                        ),
                    ),
                )
            z: Tensor = timed_until._quantitative(
                x, vectorize=vectorize, normalize=normalize
            )
        else:
            if self.left_time_bound == 0:
                timed_until: Node = And(
                        self.left_child,
                    And(
                        Eventually(
                            self.right_child,
                            left_time_bound=self.left_time_bound,
                            right_time_bound=self.right_time_bound,  # - 1,
                        ),
                        Eventually(
                            Until(self.left_child, self.right_child, unbound=True),
                            left_time_bound=self.left_time_bound,
                            right_unbound=True,
                        ),
                    ),
                )
            else:
                timed_until: Node = And(
                        Globally(
                            self.left_child,
                            left_time_bound=0,
                            right_time_bound=self.left_time_bound,
                        ),
                        And(
                            Eventually(
                                self.right_child,
                                left_time_bound=self.left_time_bound,
                                right_time_bound=self.right_time_bound,  # - 1,
                            ),
                            Eventually(
                                Until(self.left_child, self.right_child, unbound=True),
                                left_time_bound=self.left_time_bound,
                                right_unbound=True,
                            ),
                        ),
                    )
            z: Tensor = timed_until._quantitative(
                x, vectorize=vectorize, normalize=normalize
            )
        return z.cpu()
