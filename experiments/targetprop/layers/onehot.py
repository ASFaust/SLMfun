import torch
#this is a softmax-like function for target propagation
#it is used as the final classification layer in a dummy classifier that im going to train.
#it works differently: the maximal input is set to 1, and all other inputs are set to 0.
#we can define get_ft easily as the same function.

class TargetPropagationOnehot:
    def __init__(self, input_hook):
        self.input_hook = input_hook
        self.input = None  # To store input during forward pass
        self.output = None

    def __call__(self, x):
        """
        Forward pass: Applies onehot activation.
        :param x:
        :return:
        """
        with torch.no_grad():
            # Get max value and index
            max_val, max_idx = torch.max(x, dim=1, keepdim=True)
            y = torch.zeros_like(x)
            y[torch.arange(x.shape[0]), max_idx.squeeze()] = 1
            self.input = x
            self.output = y

        return y

    def get_ft(self, x_prime):
        """
        Computes the feasible target x'' for the next layer.
        :param x_prime: the target from the next layer. computed as x + dx, reflecting how the input should change.
        it has shape (batch_size, features)
        """
        return self.__call__(x_prime)

    def backward(self, y_prime):
        """
        Performs backward target propagation.
        :param y_prime: the target for the output of this layer. it has shape (batch_size, features). It should adhere to the image of this layer. It is often computed using get_ft of this layer.
        :return: Nothing, invokes the backward method of the input_hook. Computed is the target for the input_hook layer.
        """
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