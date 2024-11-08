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

    def save(self, path):
        save_dict = {
            'batch_size': self.batch_size,
            'vocab_size': self.vocab_size,
            'memory_size': self.memory_size,
            'hidden_dims': self.hidden_dim,
            'device': self.device,
            'state_dict': self.state_dict()
        }
        torch.save(save_dict, path)

    @staticmethod
    def load(path, batch_size=None, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        load_dict = torch.load(path, map_location=device)
        if batch_size is not None:
            load_dict['batch_size'] = batch_size
        net = Net(
            load_dict['batch_size'],
            load_dict['vocab_size'],
            load_dict['memory_size'],
            load_dict['hidden_dims'],
            device
        )
        net.load_state_dict(load_dict['state_dict'])
        return net