import torch

class ReLUTransformLayer:
    def forward(self, x):
        # Compute x + ReLU(x)
        relu_x = torch.relu(x)
        y = x + relu_x
        return y

    def backward(self, y_prime):
        # Compute the inverse of x + ReLU(x)
        # If x >= 0: x + x = y -> x = y / 2
        # If x < 0: x + 0 = y -> x = y
        x_prime = torch.where(y_prime > 0, y_prime / 2, y_prime)
        return x_prime
