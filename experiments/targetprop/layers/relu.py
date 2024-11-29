import torch

class TargetPropagationReLU:
    def __init__(self, input_hook):
        self.input_hook = input_hook
        self.input = None  # To store input during forward pass

    def __call__(self, x):
        """
        Forward pass: Applies ReLU activation.
        """
        with torch.no_grad():
            self.input = x  # Save input for use in backward pass
            y = torch.relu(x)
        return y

    def get_ft(self, x_prime):
        with torch.no_grad():
            ret = torch.relu(x_prime) # feasible target is the ReLU of the target itself
            # uhh, the feasible target should be normalized to 1.0
            # ret_len = torch.norm(ret, dim=1, keepdim=True)
            # ret = ret / ret_len
            return ret

    def backward(self, y_prime):
        """
        Performs backward target propagation.
        Args:
            y_prime: feasible target from the next layer (batch_size, features). should always be computed using get_ft.
        """
        with torch.no_grad():
            # Initialize target
            ret = y_prime.clone()

            # Case: input <= 0 and y_prime == 0
            mask_case = (self.input <= 0) & (y_prime == 0)
            ret[mask_case] = self.input[mask_case]

        # Call backward on input_hook with the computed target
        self.input_hook.backward(ret)
