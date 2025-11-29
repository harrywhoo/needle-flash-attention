"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = None
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = ops.matmul(X, self.weight)
        if self.bias:
            y = y + self.bias.broadcast_to(y.shape)
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        # Calculate the product of all non-batch dimensions
        feature_size = 1
        for dim in X.shape[1:]:
            feature_size *= dim
        return ops.reshape(X, (batch_size, feature_size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        res = x
        for module in self.modules:
            res = module(res)
        return res
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        m = logits.shape[0]
        k = logits.shape[1]
        y_one_hot = init.one_hot(k, y, device=logits.device, dtype=logits.dtype)
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        z_y = ops.summation(ops.multiply(logits, y_one_hot), axes=(1,))
        loss = ops.summation(log_sum_exp - z_y) / m
        return loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, feature_size = x.shape

        if self.training:
            mean = ops.summation(x, axes=(0,)) / batch_size
            mean_reshaped = ops.reshape(mean, (1, feature_size))
            mean_broadcast = ops.broadcast_to(mean_reshaped, x.shape)

            x_centered = x - mean_broadcast
            var = ops.summation(x_centered ** 2, axes=(0,)) / batch_size
            var_reshaped = ops.reshape(var, (1, feature_size))
            var_broadcast = ops.broadcast_to(var_reshaped, x.shape)

            # Update running estimates
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data

            std = (var_broadcast + self.eps) ** 0.5
            normalized = x_centered / std
        else:
            mean_reshaped = ops.reshape(self.running_mean, (1, feature_size))
            mean_broadcast = ops.broadcast_to(mean_reshaped, x.shape)
            var_reshaped = ops.reshape(self.running_var, (1, feature_size))
            var_broadcast = ops.broadcast_to(var_reshaped, x.shape)

            x_centered = x - mean_broadcast
            std = (var_broadcast + self.eps) ** 0.5
            normalized = x_centered / std

        weight_broadcast = ops.broadcast_to(ops.reshape(self.weight, (1, feature_size)), x.shape)
        bias_broadcast = ops.broadcast_to(ops.reshape(self.bias, (1, feature_size)), x.shape)

        return weight_broadcast * normalized + bias_broadcast
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, feature_size = x.shape
        mean = ops.summation(x, axes=(1,)) / feature_size
        mean_reshaped = ops.reshape(mean, (batch_size, 1))
        mean_broadcast = ops.broadcast_to(mean_reshaped, x.shape)

        x_centered = x - mean_broadcast
        var = ops.summation(x_centered ** 2, axes=(1,)) / feature_size
        var_reshaped = ops.reshape(var, (batch_size, 1))
        var_broadcast = ops.broadcast_to(var_reshaped, x.shape)

        std = (var_broadcast + self.eps) ** 0.5
        normalized = x_centered / std

        weight_broadcast = ops.broadcast_to(ops.reshape(self.weight, (1, feature_size)), x.shape)
        bias_broadcast = ops.broadcast_to(ops.reshape(self.bias, (1, feature_size)), x.shape)

        return weight_broadcast * normalized + bias_broadcast
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=(1 - self.p), dtype="float32", device=x.device)
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
