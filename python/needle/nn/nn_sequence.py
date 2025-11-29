"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # sigmoid(x) = 1 / (1 + exp(-x))
        return (1 + ops.exp(-x)) ** -1
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype

        bound = (1.0 / hidden_size) ** 0.5
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))

        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]

        if h is None:
            h = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)

        # h' = nonlinearity(X @ W_ih + h @ W_hh + bias_ih + bias_hh)
        out = X @ self.W_ih + h @ self.W_hh

        if self.bias_ih is not None:
            out = out + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))
        if self.bias_hh is not None:
            out = out + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))

        if self.nonlinearity == 'tanh':
            return ops.tanh(out)
        elif self.nonlinearity == 'relu':
            return ops.relu(out)
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

        self.rnn_cells = [RNNCell(input_size if i == 0 else hidden_size, hidden_size, bias, nonlinearity, device, dtype)
            for i in range(num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape

        # Initialize h0 if not provided
        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)

        # Split h0 into list of hidden states per layer
        h_layers = list(ops.split(h0, axis=0))

        # Split X into list of time steps
        X_steps = list(ops.split(X, axis=0))

        outputs = []

        # Process each time step
        for t in range(seq_len):
            x_t = X_steps[t]

            # Process through all layers
            for layer in range(self.num_layers):
                h_layers[layer] = self.rnn_cells[layer](x_t, h_layers[layer])
                x_t = h_layers[layer]

            outputs.append(x_t)

        output = ops.stack(outputs, axis=0)
        h_n = ops.stack(h_layers, axis=0)

        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

        bound = (1.0 / hidden_size) ** 0.5
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))

        if bias:
            self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]

        if h is None:
            h0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h

        # Compute gates
        gates = X @ self.W_ih + h0 @ self.W_hh

        if self.bias_ih is not None:
            gates = gates + self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to((bs, 4 * self.hidden_size))
        if self.bias_hh is not None:
            gates = gates + self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to((bs, 4 * self.hidden_size))

        # Split into 4 gates
        gates_list = list(ops.split(gates.reshape((bs, 4, self.hidden_size)), axis=1))
        i = Sigmoid()(gates_list[0].reshape((bs, self.hidden_size)))
        f = Sigmoid()(gates_list[1].reshape((bs, self.hidden_size)))
        g = ops.tanh(gates_list[2].reshape((bs, self.hidden_size)))
        o = Sigmoid()(gates_list[3].reshape((bs, self.hidden_size)))

        # c' = f * c + i * g
        c_out = f * c0 + i * g

        # h' = o * tanh(c')
        h_out = o * ops.tanh(c_out)

        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

        self.lstm_cells = [
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias, device, dtype)
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape

        # Initialize h and c if not provided
        if h is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h

        # Split into list of states per layer
        h_layers = list(ops.split(h0, axis=0))
        c_layers = list(ops.split(c0, axis=0))

        # Split X into list of time steps
        X_steps = list(ops.split(X, axis=0))

        outputs = []

        # Process each time step
        for t in range(seq_len):
            x_t = X_steps[t]

            # Process through all layers
            for layer in range(self.num_layers):
                h_layers[layer], c_layers[layer] = self.lstm_cells[layer](x_t, (h_layers[layer], c_layers[layer]))
                x_t = h_layers[layer]

            outputs.append(x_t)

        # Stack outputs
        output = ops.stack(outputs, axis=0)

        # Stack final states
        h_n = ops.stack(h_layers, axis=0)
        c_n = ops.stack(c_layers, axis=0)

        return output, (h_n, c_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0.0, std=1.0, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape

        # Convert indices to one-hot
        # (seq_len, bs) -> (seq_len, bs, num_embeddings)
        one_hot = init.one_hot(self.num_embeddings, x, device=self.device, dtype=self.dtype)

        # Reshape to (seq_len * bs, num_embeddings)
        one_hot_flattened = one_hot.reshape((seq_len * bs, self.num_embeddings))

        output_flat = one_hot_flattened @ self.weight

        # Reshape back to (seq_len, bs, embedding_dim)
        return output_flat.reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION