import torch


class InputHook:
    def __init__(self):
        self.input = None

    def __call__(self, x):
        self.input = x
        return x

    def get_ft(self, x_prime):
        # For the input layer, the feasible target is the target itself
        return x_prime

    def backward(self, x_prime):
        # For the input layer, you might store or process x_prime as needed
        pass
