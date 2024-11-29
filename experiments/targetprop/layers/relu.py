import torch

class TargetPropagationReLU:
    def __init__(self, input_hook):
        self.input_hook = input_hook
        self.input = None

    def __call__(self, x):
        """
        Forward pass: Applies ReLU activation.
        """
        with torch.no_grad():
            self.input = x
            y = torch.relu(x)
        return y

    def get_ft(self, x_prime):
        with torch.no_grad():
            #min_per_batch = torch.min(x_prime, dim=1, keepdim=True)[0]
            #ret = x_prime + min_per_batch #this guarantees that the target is always positive. kind of a hack lmao
            ret = torch.relu(x_prime) #this is cleaner
            return ret

    def backward(self, y_prime):
        """
        Performs backward target propagation.
        Args:
            y_prime: feasible target from the next layer (batch_size, features). should always be computed using get_ft.
        """
        with torch.no_grad():
            # Initialize target
            target = torch.zeros_like(self.input)
            target[y_prime > 0] = y_prime[y_prime > 0] #here we need to enforce that the target is correct.
            target[y_prime <= 0] = torch.clamp_max(self.input[y_prime <= 0], 0.0) #this is the same as the onehot layer.

        # Call backward on input_hook with the computed target
        self.input_hook.backward(target)
