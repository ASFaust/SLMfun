import torch
#this is a n-hot function for target propagation
#it activates the n largest values in the input and sets the rest to 0

class TargetPropagationNhot:
    def __init__(self, input_hook, n):
        self.input_hook = input_hook
        self.input = None  # To store input during forward pass
        self.output = None
        self.x_prime = None
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
        For a onehot layer, the feasible target is the same as its forward pass.
        """
        self.x_prime = x_prime
        return self.__call__(x_prime)

    def backward(self, y_prime):
        target = self.input.clone()
        #we use a swapping method to get the target
        #we swap the n largest values in the input with the n largest values in the target
        #but how do we know which ones to switch with which?
        #we compare x_prime and y_prime. later, after i got vaccinated, i will explain why this works.