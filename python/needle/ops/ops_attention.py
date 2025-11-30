"""Flash Attention operations."""

from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from ..backend_selection import array_api, BACKEND

class FlashAttention(TensorOp):
    """
    Flash Attention operation with tiling for O(N) memory complexity.
    Computes softmax(Q @ K^T / sqrt(d)) @ V using tiling and online softmax.
    """
    def __init__(self, causal: bool = False, block_m: int = 128, block_n: int = None):
        """
        Args:
            causal: If True, apply causal masking
            block_m: Block size for Q 
            block_n: Block size for K, V, defaults to block_m
        """
        self.causal = causal
        self.block_m = block_m
        self.block_n = block_n if block_n is not None else block_m

    def compute(self, Q, K, V):
        """
        Forward pass of Flash Attention.

        Args:
            Q: Query NDArray, shape (batch, num_heads, seq_len, head_dim)
            K: Key NDArray, shape (batch, num_heads, seq_len, head_dim)
            V: Value NDArray, shape (batch, num_heads, seq_len, head_dim)

        Returns:
            O: Output NDArray, shape (batch, num_heads, seq_len, head_dim)

        Note: m and l statistics need to be saved for backward pass.
        """
        
        raise NotImplementedError("FlashAttention forward not yet implemented")

    def gradient(self, out_grad, node):
        """
        Backward pass of Flash Attention.

        Args:
            out_grad: Gradient w.r.t. output O
            node: The computational graph node containing cached values

        Returns:
            Tuple of (dQ, dK, dV) gradients
        """
       
        raise NotImplementedError("FlashAttention backward not yet implemented")


def flash_attention(Q, K, V, causal: bool = False, block_m: int = 128, block_n: int = None):
    """Flash Attention with tiling."""
    return FlashAttention(causal=causal, block_m=block_m, block_n=block_n)(Q, K, V)
