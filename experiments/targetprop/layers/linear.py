import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetPropagationLinear:
    def __init__(self, in_features, out_features, input_hook, epsilon=1e-8):
        # Adjusted the weight shape to (out_features, in_features)
        self.w = torch.randn((out_features, in_features), requires_grad=False) * 0.001
        self.input_hook = input_hook
        self.y = None  # To store output during forward pass
        self.x = None  # To store input during forward pass
        self.epsilon = epsilon
        #self.acc_w = torch.zeros_like(self.w)

    def __call__(self, x):
        """
        Standard forward pass.
        """
        with torch.no_grad():
            # Adjusted to match the new weight shape
            y = torch.matmul(x, self.w.T)  # (batch_size, in_features) @ (in_features, out_features)
            self.y = y.clone()
            self.x = x.clone()
        return y

    def get_ft(self, x_prime):
        """
        Computes the feasible target x'' for the current layer.
        For a linear layer, the feasible target is the target itself.
        """
        return x_prime

    def backward(self, y_prime):
        """
        Performs the target propagation step.
        Args:
            y_prime: Target from the next layer (batch_size, out_features)
            Returns:
                x_prime: Target for the previous layer (batch_size, in_features)
        """
        with torch.no_grad():
            # Compute delta_y
            delta_y = y_prime - self.y  # (batch_size, out_features)

            # Compute delta_y_w: element-wise multiplication
            # Expand dimensions to allow broadcasting
            delta_y_exp = delta_y.unsqueeze(2)  # (batch_size, out_features, 1)
            w_exp = self.w.unsqueeze(0)  # (1, out_features, in_features)

            delta_x = delta_y_exp * w_exp  # (batch_size, out_features, in_features)

            # Compute x': mean over neurons (out_features dimension)
            x_prime = self.x + delta_x.mean(dim=1)  # (batch_size, in_features)

            # Compute feasible target x'': apply activation function
            x_double_prime = self.input_hook.get_ft(x_prime)


            # Compute y'' = <x'', w>
            y_double_prime = torch.matmul(x_double_prime, self.w.T)  # (batch_size, out_features)

            # Compute numerator: (y' - y'')
            delta_y_prime = y_prime - y_double_prime  # (batch_size, out_features)

            # Compute denominator: ||x''||^2
            x_double_prime_norm_sq = x_double_prime.pow(2).sum(dim=1, keepdim=True)  # (batch_size, 1)
            x_double_prime_norm_sq = x_double_prime_norm_sq #+ self.epsilon

            # Compute update factor
            update_factor = delta_y_prime.unsqueeze(2) / x_double_prime_norm_sq.unsqueeze(2)  # (batch_size, out_features, 1)

            # Compute delta_w
            x_double_prime_exp = x_double_prime.unsqueeze(1)  # (batch_size, 1, in_features)
            delta_w = (update_factor * x_double_prime_exp).mean(dim=0)  # (out_features, in_features)

            #self.acc_w *= 0.9
            #self.acc_w += delta_w

            self.w += delta_w

        self.input_hook.backward(x_double_prime)  # Call backward on the input hook with the feasible target


if __name__ == '__main__':
    # Define input hook
    from inputhook import InputHook
    input_hook = InputHook()

    # Define target propagation layers
    linear1 = TargetPropagationLinear(2, 3, input_hook)

