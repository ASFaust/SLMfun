import torch
#this is a binary activation function for target propagation

class TargetPropagationBinary:
    def __init__(self, input_hook):
        self.input_hook = input_hook
        self.input = None  # To store input during forward pass
        self.output = None

    def __call__(self, x):
        """
        Forward pass: Applies binary activation.
        :param x:
        :return:
        """
        with torch.no_grad():
            self.input = x
            y = torch.zeros_like(x)
            x -= x.mean(dim=1, keepdim=True) #we center the input, always
            y[x > 0] = 1
            self.output = y
        return y

    def get_ft(self, x_prime):
        """
        Computes the feasible target x'' for the current layer.
        For a binary layer, the feasible target is the same as its forward pass.
        maybe we should do batchnorm over x_prime first.
        """

        return self.__call__(x_prime)

    def backward(self, y_prime):
        target = self.input.clone()
        #we enforce target[x > 0] >= 1 and target[x <= 0] <= -1
        target[y_prime == 1] = torch.clamp_min(target[y_prime == 1], 1.0)
        target[y_prime == 0] = torch.clamp_max(target[y_prime == 0], 0.0)

        self.input_hook.backward(target)