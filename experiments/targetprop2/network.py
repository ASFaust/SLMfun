import torch
from activation1 import ReLUTransformLayer
from bias import TargetPropBias
from linear import TargetPropLinear

class TargetPropNetwork:
    def __init__(self, layer_sizes, use_bias=True):
        self.layers = []
        self.use_bias = use_bias

        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]

            # If using bias, increase input dimension by 1
            if self.use_bias:
                self.layers.append(TargetPropBias())
                input_dim += 1

            # Add linear layer
            self.layers.append(TargetPropLinear(input_dim, output_dim))

            # Add ReLUTransformLayer
            self.layers.append(ReLUTransformLayer())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_prime):
        for layer in reversed(self.layers):
            y_prime = layer.backward(y_prime)
        return y_prime
