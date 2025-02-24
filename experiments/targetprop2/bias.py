import torch

class TargetPropBias:
    def forward(self, x):
        with torch.no_grad():
            # Create a tensor of ones with the same shape as x except for the last dimension
            ones = torch.ones(*x.shape[:-1], 1, device=x.device)
            # Concatenate the bias term along the last dimension
            x_with_bias = torch.cat([x, ones], dim=-1)
            return x_with_bias

    def backward(self, y_prime):
        with torch.no_grad():
            # Drop the last dimension (the bias information)
            x_prime = y_prime[..., :-1]
            return x_prime
