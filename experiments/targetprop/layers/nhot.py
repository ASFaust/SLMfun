import torch
#this is a n-hot function for target propagation
#it activates the n largest values in the input and sets the rest to 0

class TargetPropagationNhot:
    def __init__(self, input_hook, n):
        self.input_hook = input_hook
        self.input = None  # To store input during forward pass
        self.output = None
        self.n = n

    def __call__(self, x):
        """
        Forward pass: Applies onehot activation.
        :param x:
        :return:
        """
        with torch.no_grad():
            self.input = x
            y = torch.zeros_like(x)
            _, max_idx = torch.topk(x, self.n, dim=1)
            y[torch.arange(x.shape[0])[:, None], max_idx] = 1
            self.output = y
        return y

    def get_ft(self, x_prime):
        """
        Computes the feasible target x'' for the current layer.
        For a nhot layer, the feasible target is the same as its forward pass.
        """
        return self.__call__(x_prime)

    def backward(self, y_prime):
        #we do the same as in onehot, but with the n biggest values instead of the top 1 value
        #for reference, here is the code for onehot:
        """
            def backward(self, y_prime):
        with torch.no_grad():
            # Start with a clone of the input
            ret = self.input.clone()

            # For the target class (where y_prime == 1), ensure ret >= 1
            target_indices = y_prime == 1
            ret[target_indices] = torch.clamp_min(ret[target_indices], 1.0)

            # For all other classes (where y_prime == 0), ensure ret <= 0
            non_target_indices = y_prime == 0
            ret[non_target_indices] = torch.clamp_max(ret[non_target_indices], 0.0)

            # Pass the modified ret to the previous layer
            self.input_hook.backward(ret)
        """
        target = self.input.clone()

        target_indices = y_prime == 1
        target[target_indices] = torch.clamp_min(target[target_indices], 1.0)

        non_target_indices = y_prime == 0
        target[non_target_indices] = torch.clamp_max(target[non_target_indices], 0.0)

        self.input_hook.backward(target)