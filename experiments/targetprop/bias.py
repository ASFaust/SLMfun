import torch
import torch.nn as nn

class TargetPropagationBias:
    def __init__(self, input_hook):
        self.input_hook = input_hook

    def __call__(self, x):
        """
        Standard forward pass. All it does is concat a 1 to the input.
        """
        with torch.no_grad():
            y = torch.cat((x, torch.ones((x.shape[0], 1), device=x.device)), dim=1)
        return y

    def get_ft(self, x_prime):
        """
        Compute the feasible target. for the first n elements of x_prime, return self.input_hook.get_ft(x_prime).
        for the last element, return 1.
        """
        with torch.no_grad():
            ret = self.input_hook.get_ft(x_prime[:, :-1])
            return torch.cat((ret, torch.ones((ret.shape[0], 1), device=ret.device)), dim=1)

    def backward(self, y_prime):
        """
        Performs backward target propagation.
        Args:
            y_prime: Target from the next layer (batch_size, n+1)
        backward targets are always feasible.
        """
        with torch.no_grad():
            self.input_hook.backward(y_prime[:, :-1])
