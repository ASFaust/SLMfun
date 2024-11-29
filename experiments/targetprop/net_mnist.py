from layers.bias import TargetPropagationBias
from layers.linear import TargetPropagationLinear
from layers.relu import TargetPropagationReLU
from layers.inputhook import InputHook
from layers.onehot import TargetPropagationOnehot
from layers.binary import TargetPropagationBinary

class Net:
    def __init__(self):
        self.input_hook = InputHook()
        self.bias1 = TargetPropagationBias(self.input_hook)
        self.linear1 = TargetPropagationLinear(28*28 + 1, 32, self.bias1)
        self.relu = TargetPropagationReLU(self.linear1)
        self.bias2 = TargetPropagationBias(self.relu)
        self.linear2 = TargetPropagationLinear(32 + 1, 10, self.bias2)
        self.onehot = TargetPropagationOnehot(self.linear2)

    def forward(self, x):
        x = self.input_hook(x)
        x = self.bias1(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.bias2(x)
        x = self.linear2(x)
        x = self.onehot(x)
        return x

    def backward(self, target):
        """
        Performs backward target propagation.
        :param target: should be a onehot vector of the MNIST label, of shape (batch_size, 10)
        :return: None
        """
        self.onehot.backward(target) #since all others are hooked onto this, we can just call backward on this one.

