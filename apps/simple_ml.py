"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # labels
    with gzip.open(label_filename, 'rb') as label_file:
        magic = int.from_bytes(label_file.read(4), 'big')
        if magic != 2049:
            raise ValueError("Invalid magic number in label file")
        num_labels = int.from_bytes(label_file.read(4), 'big')
        labels_bytes = label_file.read(num_labels)
        y = np.frombuffer(labels_bytes, dtype=np.uint8)

    # Images
    with gzip.open(image_filename, 'rb') as image_file:
        magic = int.from_bytes(image_file.read(4), 'big')
        if magic != 2051:
            raise ValueError("Invalid magic number in image file")
        num_images = int.from_bytes(image_file.read(4), 'big')
        rows = int.from_bytes(image_file.read(4), 'big')
        cols = int.from_bytes(image_file.read(4), 'big')
        pixels = num_images * rows * cols
        image_bytes = image_file.read(pixels)
        X = np.frombuffer(image_bytes, dtype=np.uint8)
        # reshape dimensions and normalize to 0-1
        X = X.reshape(num_images, rows * cols).astype(np.float32) / 255.0

    return (X, y)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    log_sum_exp = ndl.log(ndl.summation(ndl.exp(Z), axes=1))
    true_logits = ndl.summation(Z * y_one_hot, axes=1)
    loss_per_example = log_sum_exp - true_logits
    batch_size = Z.shape[0]
    average_loss = ndl.summation(loss_per_example) / batch_size
    return average_loss
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    n = X.shape[0]
    d = W1.shape[1]
    k = W2.shape[1]

    for i in range(0, n, batch):
        X_batch = X[i:i+batch]
        y_batch = y[i:i+batch]
        m = X_batch.shape[0]

        X_batch_tensor = ndl.Tensor(X_batch, requires_grad=False)
        one_hot = np.zeros((m, k), dtype=X_batch.dtype)
        one_hot[np.arange(m), y_batch] = 1.0

        y_batch_tensor = ndl.Tensor(one_hot, requires_grad=False)

        # forward
        Z1 = ndl.relu(ndl.matmul(X_batch_tensor, W1))
        logits = ndl.matmul(Z1, W2)

        lse = ndl.log(ndl.summation(ndl.exp(logits), axes=(1,)))
        true_logit = ndl.summation(ndl.multiply(logits, y_batch_tensor), axes=(1,))
        loss = ndl.summation(lse - true_logit) / m

        # backward
        loss.backward()

        # update weights
        W1_numpy = W1.numpy() - lr * W1.grad.numpy()
        W2_numpy = W2.numpy() - lr * W2.grad.numpy()
        W1 = ndl.Tensor(W1_numpy, requires_grad=True)
        W2 = ndl.Tensor(W2_numpy, requires_grad=True)

    return W1, W2
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    if opt is not None:
        model.train()
    else:
        model.eval()

    for batch in dataloader:
        X, y = batch

        # Forward pass
        logits = model(X)
        loss = loss_fn(logits, y)

        # Compute accuracy
        predictions = np.argmax(logits.numpy(), axis=1)
        correct = np.sum(predictions == y.numpy())

        total_loss += loss.numpy() * X.shape[0]
        total_correct += correct
        total_samples += X.shape[0]

        # Backward pass and optimization (if training)
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = loss_fn()

    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_function, opt)

    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn_instance = loss_fn()
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn_instance, opt=None)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    nbatch = len(data)
    h = None

    for i in range(0, nbatch - 1, seq_len):
        X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)

        logits, h = model(X, h)

        if isinstance(h, tuple):
            h = (h[0].detach(), h[1].detach())
        else:
            h = h.detach()

        loss = loss_fn(logits, y)

        total_loss += loss.numpy() * y.shape[0]
        total_correct += (logits.numpy().argmax(axis=1) == y.numpy()).sum()
        total_samples += y.shape[0]

        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = loss_fn()

    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_function, opt, clip, device, dtype)

    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_function = loss_fn()
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_function, opt=None, device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
