import torch
import torch.nn as nn
import math

from DenseNetwork import DenseNetwork

class Net(nn.Module):
    def __init__(self,
                 batch_size,
                 vocab_size,
                 memory_size,
                 hidden_dims,
                 device='cpu'):
        super(Net, self).__init__()
        # Assign parameters
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.memory_size = memory_size
        self.hidden_dim = hidden_dims
        self.device = device

        # Initialize memory, timings, and surprise tensors
        self.memory = torch.zeros((self.batch_size, self.memory_size, self.vocab_size), device=self.device)

        input_size = self.vocab_size * self.memory_size

        self.pred_mlp = DenseNetwork(input_size, hidden_dims, vocab_size, "gated", True, device)

    def forward(self, x):
        #roll the memory
        self.memory = torch.roll(self.memory, shifts=1, dims=1)
        self.memory[:, 0, :] = x
        #flatten the memory
        memory_flat = self.memory.view(self.batch_size, -1)
        pred_logits = self.pred_mlp(memory_flat)
        return pred_logits

    def reset(self):
        self.memory = torch.zeros((self.batch_size, self.memory_size, self.vocab_size), device=self.device)

