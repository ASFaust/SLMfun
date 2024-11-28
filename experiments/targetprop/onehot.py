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
        Computes the feasible target x'' for the current layer.
        For a onehot layer, the feasible target is the same as its forward pass.
        """
        return self.__call__(x_prime)

    def backward(self, y_prime):

        with torch.no_grad():
            ret = self.input.clone()
            ret[self.output == 1] = self.input[y_prime == 1]
            ret[y_prime == 1] = self.input[self.output == 1]
            self.input_hook.backward(ret)
