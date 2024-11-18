import torch

"""
we will not use backprop functionality of torch, so we will not need to store gradients
we will write our own explicit target-propagation algorithm
so we will not need to store the weights as parameters
"""


class InputLinear:
    def __init__(self, input_size, output_size, device):
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.weights = torch.zeros((self.input_size, self.output_size), device=self.device, requires_grad=False)
        self.dw = torch.zeros((self.input_size, self.output_size), device=self.device, requires_grad=False)
        self.v = None
        self.x = None

    def forward(self, x):
        with torch.no_grad():
            self.v = torch.matmul(x, self.weights)  #save v for backpropagation
            self.x = x.clone()  #save x for backpropagation
        return self.v

    def backward(self, v_hat):
        #v_hat has shape (batch_size, output_size)
        #self.x has shape (batch_size, input_size)
        #use the following formula: dw = dv/|x|² * x.
        #where dv is v_hat - v
        #and dw is w_hat - w, where we want to obtain w_hat
        with torch.no_grad():
            dv = v_hat - self.v
            dw = torch.matmul(self.x.t(), dv) / (torch.norm(self.x) ** 2)
            #we average dw across the batch dimension and save it first separately
            self.dw = torch.mean(dw, dim=0) #has shape (input_size, output_size), same as self.weights
        return None #Input linear layer does not create gradients for the previous layer, as it assumes it is the input layer

