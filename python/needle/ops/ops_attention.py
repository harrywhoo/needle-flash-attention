"""Attention operations."""

from typing import Optional
import numpy as np
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import needle.backend_ndarray.ndarray as ndarray
from .nn_basic import Dropout 

from ..backend_selection import array_api, BACKEND

# Helper functions for regular attention 
def create_causal_mask(i: int, j: int, device) -> NDArray:
    """"
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
    """
    mask = -np.finfo(np.float32).max * np.triu(
        np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)
    return ndarray.array(mask, device=device)

def batched_matmul(a: Tensor, b_transpose: Tensor) -> Tensor:
    """
    batched matrix multiplication;
    """
    a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
    a = a.reshape(a_shape)

    b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
    b_transpose = b_transpose.reshape(b_transpose_shape)

    broadcast_shape = list(a_shape)
    broadcast_shape[-2] = b_transpose_shape[-2]
    a = a.broadcast_to(broadcast_shape)

    broadcast_shape = list(b_transpose_shape)
    broadcast_shape[-3] = a_shape[-3]
    b_transpose = b_transpose.broadcast_to(broadcast_shape)

    return (a * b_transpose).sum(len(a.shape) - 1)

def attention_softmax(logits: Tensor) -> Tensor:
    """
    Softmax function for attention
    """
    max_val = Tensor(
        logits.realize_cached_data().max(axis=3),
        device=logits.device,
        dtype=logits.dtype,
        requires_grad=False
    )

    max_val = max_val.reshape((*logits.shape[:-1], 1))
    max_val = max_val.broadcast_to(logits.shape)

    probs = ops.exp(logits - max_val)

    denom = probs.sum(axes=3)
    denom = denom.reshape((*logits.shape[:-1], 1))
    denom = denom.broadcast_to(logits.shape)

    return probs / denom

###############################################################################
# Regular Attention (Baseline Implementation)
###############################################################################

def regular_attention(
    Q: Tensor, K: Tensor, V: Tensor,
    causal: bool = False,
    dropout: Dropout = None
):
    """
    Standard attention implementation using autograd.

    Args:
        Q: Query tensor of shape (batch, num_heads, N, d)
        K: Key tensor of shape (batch, num_heads, N, d)
        V: Value tensor of shape (batch, num_heads, N, d)
        causal: If True, apply causal masking
        dropout: Optional Dropout module to apply to attention weights

    Returns:
        Tuple of (output, probs):
            output: Tensor of shape (batch, num_heads, N, d)
            probs: Attention probabilities of shape (batch, num_heads, N, N) after dropout
    """
    batch_size, num_heads, queries_len, q_dim = Q.shape # queries_len = , q_dim = d 
    _, _, keys_values_len, _ = K.shape

    # Compute attention scores: Q @ K^T / sqrt(d)
    weight_matrix = batched_matmul(Q, K) / (q_dim ** 0.5)  # (batch, num_heads, N, N)

    # Apply causal mask if needed
    if causal:
        mask = create_causal_mask(queries_len, keys_values_len, weight_matrix.device)
        mask_tensor = Tensor(mask, device=weight_matrix.device, dtype=weight_matrix.dtype)
        mask_tensor = mask_tensor.broadcast_to(weight_matrix.shape)
        weight_matrix = weight_matrix + mask_tensor

    # Softmax and dropout
    attn_probs = attention_softmax(weight_matrix)

    if dropout is not None:
        probs = dropout(attn_probs)
    else:
        probs = attn_probs

    # Apply attention to values: attn @ V
    V_T = ops.transpose(V, axes=(2, 3))  # (batch, num_heads, d, N)
    result = batched_matmul(probs, V_T)  # (batch, num_heads, N, d)

    return result, probs


class FlashAttention(TensorOp):
    """
    Flash Attention operation with tiling for O(N) memory complexity.
    Computes softmax(Q @ K^T / sqrt(d)) @ V using tiling and online softmax.

    Following Flash Attention paper notation:
        Br = row block size (for Q)
        Bc = column block size (for K, V)
    """
    def __init__(self, causal: bool = False, Bc: int = 128, Br: int = None):
        """
        Args:
            causal: If True, apply causal masking
            Bc: Column block size for K, V (paper notation)
            Br: Row block size for Q (defaults to Bc if None)
        """
        self.causal = causal
        self.Bc = Bc
        self.Br = Br if Br is not None else Bc

    def compute(self, Q, K, V):
        """
        Forward pass of Flash Attention.

        Args:
            Q: Query NDArray, shape (batch, num_heads, N, d)
            K: Key NDArray, shape (batch, num_heads, N, d)
            V: Value NDArray, shape (batch, num_heads, N, d)

        Returns:
            O: Output NDArray, shape (batch, num_heads, N, d)

        Note: m and l statistics are cached for backward pass.
        """
        batch_size, num_heads, N, d = Q.shape # N = seq_len, d = head_dim

        # Create output arrays
        out = array_api.NDArray.make((batch_size, num_heads, N, d), device=Q.device)
        # m and l are statistics needed for backward pass (shape: batch, num_heads, N)

        # m is the block-wise max value of attention scores 
        m = array_api.NDArray.make((batch_size, num_heads, N), device=Q.device)
        # l is the block-wise normalization terms (denominator) for softmax 
        l = array_api.NDArray.make((batch_size, num_heads, N), device=Q.device)

        Q.device.flash_attention_forward(
            Q.compact()._handle,
            K.compact()._handle,
            V.compact()._handle,
            out._handle,
            m._handle,
            l._handle,
            batch_size,
            num_heads,
            N, # seq_len
            d, # head_dim
            self.Bc,
            self.Br,
            self.causal
        )

        # Store m, l, and output for backward pass
        self.m_cache = m
        self.l_cache = l
        self.out_cache = out

        return out

    def gradient(self, out_grad, node):
        """
        Backward pass of Flash Attention.

        Args:
            out_grad: Gradient w.r.t. output O (dO)
            node: The computational graph node containing cached values

        Returns:
            Tuple of (dQ, dK, dV) gradients
        """
        Q, K, V = node.inputs

        # Get NDArrays from Tensors
        Q_data = Q.realize_cached_data()
        K_data = K.realize_cached_data()
        V_data = V.realize_cached_data()
        out_grad_data = out_grad.realize_cached_data()

        batch_size, num_heads, N, d = Q_data.shape

        # Create output gradient arrays
        dQ = array_api.NDArray.make((batch_size, num_heads, N, d), device=Q_data.device)
        dK = array_api.NDArray.make((batch_size, num_heads, N, d), device=K_data.device)
        dV = array_api.NDArray.make((batch_size, num_heads, N, d), device=V_data.device)

        # Call backend implementation
        Q_data.device.flash_attention_backward(
            out_grad_data.compact()._handle, #dO
            Q_data.compact()._handle,
            K_data.compact()._handle,
            V_data.compact()._handle,
            self.out_cache._handle, # O (forward output)
            self.m_cache._handle,
            self.l_cache._handle,
            dQ._handle,
            dK._handle,
            dV._handle,
            batch_size,
            num_heads,
            N, 
            d,
            self.Bc,
            self.Br,
            self.causal
        )

        return Tensor(dQ, device=Q.device, dtype=Q.dtype), \
               Tensor(dK, device=K.device, dtype=K.dtype), \
               Tensor(dV, device=V.device, dtype=V.dtype)

# Wrapper function for flash attention 
def flash_attention(Q, K, V, causal: bool = False, Bc: int = 128, Br: int = None):
    return FlashAttention(causal=causal, Bc=Bc, Br=Br)(Q, K, V)
