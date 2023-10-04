import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))

import torch
from DenseNetwork import Dense
from StimulusLayer import StimulusLayer

class Net(torch.nn.Module):
    def __init__(self, batch_size, hidden_size, n_layers, device):
        #each layer is a dense layer and a stimulus layer. there is a final dense layer at the end.
        super(Net, self).__init__()
        self.input_size = 256
        self.output_size = 256
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.device = device
        self.layers = torch.nn.ModuleList()
        self.layers.append(Dense(self.input_size, hidden_size, 1, device))
        self.layers.append(StimulusLayer(batch_size, hidden_size, hidden_size))
        for i in range(n_layers-1):
            self.layers.append(Dense(hidden_size, hidden_size, 1, device))
            self.layers.append(StimulusLayer(batch_size, hidden_size, hidden_size))
        self.layers.append(Dense(hidden_size, self.output_size, 1, device))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def detach_state(self):
        for layer in self.layers:
            if hasattr(layer, 'detach_state'):
                layer.detach_state()

    def reset(self):
        for layer in self.layers:
            if hasattr(layer, 'reset'):
                layer.reset()
