import torch


class InputHook:
    def __init__(self):
        self.input = None

    def __call__(self, x):
        self.input = x
        return x

    def get_ft(self, x_prime):
        # For the input layer, the feasible target is the input. x_prime is an updated target, but we can't abide by it.
        return self.input

    def backward(self, x_prime):
        pass
