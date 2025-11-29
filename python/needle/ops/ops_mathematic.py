"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs

        # For z = a^b:
        # dz/da = b * a^(b-1)
        # dz/db = a^b * ln(a)

        b_minus_1 = add_scalar(b, -1)
        a_power_b_minus_1 = power(a, b_minus_1)
        grad_a = multiply(out_grad, multiply(b, a_power_b_minus_1))

        a_power_b = node  # This is the output a^b
        grad_b = multiply(out_grad, multiply(a_power_b, log(a)))

        return grad_a, grad_b
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return mul_scalar(multiply(out_grad, power_scalar(a, self.scalar - 1)), self.scalar)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = divide(out_grad, b)
        grad_b = negate(multiply(out_grad, divide(a, power_scalar(b, 2))))
        return grad_a, grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes: # specified 
            axis1, axis2 = self.axes
        else: # swap last two axes by default
            axis1, axis2 = a.ndim - 2, a.ndim - 1

        if BACKEND == "nd":
            # NDArray uses permute
            axes_list = list(range(a.ndim))
            axes_list[axis1], axes_list[axis2] = axes_list[axis2], axes_list[axis1]
            return a.permute(tuple(axes_list))
        else:
            # numpy uses swapaxes
            return array_api.swapaxes(a, axis1, axis2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact() if hasattr(a, 'compact') else a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


def unbroadcast(x: Tensor, target_shape: tuple):
    in_shape = target_shape
    out_shape = x.shape

    dims_added = len(out_shape) - len(in_shape)
    axes = []

    # add the dimensions that were added
    for i in range(dims_added):
        axes.append(i)

    # dimensions that were broadcasted (originally 1, now > 1)
    for i in range(len(in_shape)):
        if in_shape[i] == 1 and out_shape[dims_added + i] > 1:
            axes.append(dims_added + i)

    if len(axes) > 0:
        return reshape(summation(x, tuple(axes)), in_shape)
    else:
        return reshape(x, in_shape)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        return unbroadcast(out_grad, in_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, tuple) and len(self.axes) > 1:
            # Need to reduce multiple axes one at a time
            axes_list = sorted(self.axes, reverse=True)
            result = a
            for axis in axes_list:
                result = array_api.sum(result, axis)
            return result
        else:
            return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]

        input_shape = a.shape

        # No axes specified
        if self.axes is None:
            target_shape = [1] * len(input_shape)
        else:
        # If axes were specified, add back the reduced dimensions
            # handle axes being int or tuple
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            target_shape = list(input_shape)
            for ax in axes:
                target_shape[ax] = 1

        target_shape = tuple(target_shape)
        return broadcast_to(reshape(out_grad, target_shape), input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = matmul(out_grad, transpose(b, (-1, -2)))
        grad_b = matmul(transpose(a, (-1, -2)), out_grad)

        grad_a = unbroadcast(grad_a, a.shape)
        grad_b = unbroadcast(grad_b, b.shape)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # The derivative of exp(x) is exp(x), which is the output of this node
        # Use the output directly instead of recomputing exp(input)
        return multiply(out_grad, node)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        a_data = a.realize_cached_data()
        mask = a_data > 0
        return out_grad * Tensor(mask, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # d/dx tanh(x) = 1 - tanh^2(x)
        # We can reuse the output node which is tanh(x)
        tanh_x = node
        return out_grad * add_scalar(negate(tanh_x ** 2), 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # args is a tuple of NDArrays or numpy arrays
        n = len(args)
        shape = args[0].shape
        # Make sure all tensors have same shape 
        for arr in args:
            assert arr.shape == shape, "All arrays must have the same shape"

        # Create output shape: insert new dimension at axis
        new_shape = list(shape)
        new_shape.insert(self.axis, n)
        new_shape = tuple(new_shape)

        # Empty output array 
        if BACKEND == "nd":
            # NDArray backend
            res = array_api.empty(new_shape, device=args[0].device)
        else:
            # numpy backend
            res = array_api.empty(new_shape)

        # Fill output array
        for i, arr in enumerate(args):
            slices = [slice(None)] * len(new_shape)
            slices[self.axis] = i
            res[tuple(slices)] = arr

        return res.compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Split the gradient back along the axis we stacked on
        return (split(out_grad, self.axis),)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        # Split A along self.axis
        n = A.shape[self.axis]
        res = []

        # Build new shape without the split axis
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        new_shape = tuple(new_shape)

        for i in range(n):
            # Create slice for this position along the axis
            slices = [slice(None)] * A.ndim
            slices[self.axis] = i
            # Extract the slice and reshape to remove the axis
            arr = A[tuple(slices)]
            # Use array_api.reshape which works for both numpy and NDArray
            arr = array_api.reshape(arr.compact(), new_shape)
            res.append(arr)

        return tuple(res)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Stack the gradients back together
        return (stack(out_grad, self.axis),)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a

        # New shape
        new_shape = list(a.shape)
        for axis in self.axes:
            # Skip axes that are out of bounds
            if axis < len(new_shape):
                new_shape[axis] = a.shape[axis] * (self.dilation + 1)
        new_shape = tuple(new_shape)

        # Create output array filled with zeros
        if BACKEND == "nd":
            out = array_api.full(new_shape, 0.0, device=a.device)
        else:
            out = array_api.full(new_shape, 0.0)

        # Place original values at dilated positions
        slices = [slice(None)] * a.ndim
        for axis in self.axes:
            # Skip axes that are out of bounds
            if axis < a.ndim:
                slices[axis] = slice(None, None, self.dilation + 1)

        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a

        # Create slices to get values at dilated positions
        slices = [slice(None)] * a.ndim
        for axis in self.axes:
            # Skip axes that are out of bounds
            if axis < a.ndim:
                slices[axis] = slice(None, None, self.dilation + 1)

        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # A: (N, H, W, C_in) 
        # B is weight: (K, K, C_in, C_out)
        
        # padding 
        if self.padding > 0:
            # dont pad N and C
            A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        A = A.compact()
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape

        # Compute output dimensions
        out_h = (H - K) // self.stride + 1
        out_w = (W - K) // self.stride + 1

        Ns, Hs, Ws, Cs = A.strides # from lecture ipynb 
        inner_dim = K * K * C_in

        im2col = array_api.NDArray.make(
            shape=(N, out_h, out_w, K, K, C_in),
            strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs),
            device=A.device,
            handle=A._handle,
            offset=A._offset
        ).compact().reshape((N * out_h * out_w, inner_dim)) # same as (-1, inner_dim)


        # Reshape weight 
        weight_reshaped = B.compact().reshape((inner_dim, C_out))

        out = im2col @ weight_reshaped

        # Reshape back to N H W C_out
        return out.compact().reshape((N, out_h, out_w, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs

        # X: N H W C_in
        # W: K K C_in C_out
        # out: N out_h out_w C_out
        # out_grad: N out_h out_w C_out

        # grad_X = out_grad @ W.T = conv(out_grad, W_flipped)
        # grad_W = X.T @ out_grad = conv(X, out_grad)
        K = W.shape[0]
        # grad_X
        if self.stride > 1:
            out_grad_dilated = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)
        else:
            out_grad_dilated = out_grad

        # flip weight to get W.T
        W_flipped = flip(W, axes=(0, 1))

        # Step 3: Transpose weight to swap input/output channels
        W_flipped = transpose(W_flipped, axes=(2, 3))

        padding = K - 1 - self.padding
        grad_X = conv(out_grad_dilated, W_flipped, stride=1, padding=padding)

        # grad_W
        X_permuted = transpose(X, axes=(0, 3))
        out_grad_permuted = transpose(transpose(out_grad_dilated, axes=(0, 1)), axes=(1, 2))

        grad_W = conv(X_permuted, out_grad_permuted, stride=1, padding=self.padding)

        grad_W = transpose(transpose(grad_W, axes=(0, 1)), axes=(1, 2))

        return grad_X, grad_W

        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


