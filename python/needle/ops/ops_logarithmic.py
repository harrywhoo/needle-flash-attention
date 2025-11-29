from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=1, keepdims=True)
        log_sum_exp = array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=1, keepdims=True)) + max_z
        return Z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        log_softmax_output = node
        softmax = exp(log_softmax_output)
        sum_grad = summation(out_grad, axes=(1,))
        sum_grad_reshaped = reshape(sum_grad, (z.shape[0], 1))
        return out_grad - multiply(softmax, sum_grad_reshaped)
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_keepdims = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z = array_api.max(Z, axis=self.axes, keepdims=False)

        # If axes=None, reshape max_z_keepdims 
        # broadcast needs proper dimensions
        if self.axes is None:
            # Need to reshape from (1,) to (1, 1, ..., 1) with len(Z.shape) dimensions
            keepdim_shape = tuple([1] * len(Z.shape))
            max_z_keepdims = array_api.reshape(max_z_keepdims, keepdim_shape)

        # Broadcast max_z_keepdims to Z's shape for subtraction
        exp_z = array_api.exp(Z - array_api.broadcast_to(max_z_keepdims, Z.shape))
        sum_exp_z = array_api.sum(exp_z, axis=self.axes)
        return array_api.log(sum_exp_z) + max_z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        z_data = z.realize_cached_data()
        max_z_data = array_api.max(z_data, axis=self.axes, keepdims=True)

        # If axes=None, reshape max_z_data to have proper dimensions for broadcasting
        if self.axes is None:
            keepdim_shape = tuple([1] * len(z.shape))
            max_z_data = array_api.reshape(max_z_data, keepdim_shape)

        # Broadcast max_z to z's shape for subtraction
        max_z_broadcast = array_api.broadcast_to(max_z_data, z.shape)
        max_z = Tensor(max_z_broadcast, device=z.device)
        exp_z = exp(z - max_z)
        sum_exp_z = summation(exp_z, axes=self.axes)

        # Reshape for broadcasting
        grad_shape = list(z.shape)
        if self.axes is not None:
            axes = (self.axes,) if not isinstance(self.axes, tuple) else self.axes
            for axis in axes:
                grad_shape[axis] = 1
        else:
            grad_shape = [1] * len(grad_shape)

        # Ensure tensors are compact before reshaping
        out_grad_compact = Tensor(out_grad.realize_cached_data().compact(), device=out_grad.device)
        sum_exp_z_compact = Tensor(sum_exp_z.realize_cached_data().compact(), device=sum_exp_z.device)

        out_grad_reshaped = reshape(out_grad_compact, grad_shape)
        sum_exp_z_reshaped = reshape(sum_exp_z_compact, grad_shape)

        # Broadcast to exp_z shape for division (ND backend needs explicit broadcasting)
        out_grad_broadcast = broadcast_to(out_grad_reshaped, z_data.shape)
        sum_exp_z_broadcast = broadcast_to(sum_exp_z_reshaped, z_data.shape)

        return multiply(out_grad_broadcast, divide(exp_z, sum_exp_z_broadcast))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

