import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ConvBN(ndl.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, device=None, dtype="float32"):
        super().__init__()
        self.conv = nn.Conv(in_channels, out_channels, kernel_size, stride=stride, device=device, dtype=dtype)
        self.bn = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.device = device
        self.dtype = dtype

        self.conv1 = ConvBN(3, 16, kernel_size=7, stride=4, device=device, dtype=dtype)
        self.conv2 = ConvBN(16, 32, kernel_size=3, stride=2, device=device, dtype=dtype)

        # first residual 
        self.conv3 = ConvBN(32, 32, kernel_size=3, stride=1, device=device, dtype=dtype)
        self.conv4 = ConvBN(32, 32, kernel_size=3, stride=1, device=device, dtype=dtype)

        # More ConvBN layers
        self.conv5 = ConvBN(32, 64, kernel_size=3, stride=2, device=device, dtype=dtype)
        self.conv6 = ConvBN(64, 128, kernel_size=3, stride=2, device=device, dtype=dtype)

        # Second residual
        self.conv7 = ConvBN(128, 128, kernel_size=3, stride=1, device=device, dtype=dtype)
        self.conv8 = ConvBN(128, 128, kernel_size=3, stride=1, device=device, dtype=dtype)

        # Classifier
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        # Input: (N, 3, 32, 32)
        x = self.conv1(x)    
        x = self.conv2(x)

        # First residual block
        identity = x
        x = self.conv3(x)
        x = self.conv4(x) 
        x = x + identity # Residual 

        x = self.conv5(x)    
        x = self.conv6(x)

        # Second residual block
        identity = x
        x = self.conv7(x)
        x = self.conv8(x)
        x = x + identity # Residual 

        # Classifier
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x) 

        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_model = seq_model
        self.device = device
        self.dtype = dtype

        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)

        if seq_model == 'rnn':
            self.seq = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
            self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
            self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        elif seq_model == 'transformer':
            self.seq = nn.Transformer(embedding_size, hidden_size, num_layers,
                                      sequence_len=seq_len, device=device, dtype=dtype)
            self.linear = nn.Linear(embedding_size, output_size, device=device, dtype=dtype)
        else:
            raise ValueError(f"seq_model must be 'rnn', 'lstm', or 'transformer', got {seq_model}")
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape

        embeds = self.embedding(x)

        if self.seq_model == 'transformer':
            embeds = ndl.ops.transpose(embeds, axes=(0, 1))
            seq_out, h_out = self.seq(embeds, h)
            seq_out = ndl.ops.transpose(seq_out, axes=(0, 1))
            seq_out_flat = seq_out.reshape((seq_len * bs, self.embedding_size))
        else:
            seq_out, h_out = self.seq(embeds, h)
            seq_out_flat = seq_out.reshape((seq_len * bs, self.hidden_size))

        out = self.linear(seq_out_flat)

        return out, h_out
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
