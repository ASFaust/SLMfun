import torch

class TargetPropLinear:
    def __init__(self, in_dim, out_dim):
        self.weight = torch.randn((out_dim, in_dim), requires_grad=False) * 0.01
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.last_x = None
        self.last_y = None

    def forward(self, x):
        with torch.no_grad():
            self.last_x = x
            y = torch.matmul(x, self.weight.T)
            self.last_y = y
            return y

    def backward(self, y_prime):
        with torch.no_grad():
            #y_prime has shape (batch_size, out_dim)
            #self.last_x has shape (batch_size, in_dim)
            #self.weight has shape (out_dim, in_dim)

            #we compute the update to the weight matrix as the mean over the batch of (x * (dy/||x||^2))
            #where dy = y_prime - y
            dy = y_prime - self.last_y #shape (batch_size, out_dim)
            #||x||^2 = <x, x>, so we compute the dot product of x with itself
            x_norm = (self.last_x * self.last_x).sum(dim=1) #shape (batch_size)
            #compute the update
            x_norm = x_norm.unsqueeze(1) #shape (batch_size, 1)
            factor = dy / x_norm #shape (batch_size, out_dim)
            dw = torch.matmul(factor.T, self.last_x) / self.last_x.shape[0] #shape (out_dim, in_dim)

            self.weight += dw

            #recompute the output
            y = torch.matmul(self.last_x, self.weight.T)
            self.last_y = y
            dy = y_prime - y
            #then we create the target value for x in the previous layer: factor = dy / ||w||^2
            #and this is average
            #so we compute the dot product of each neuron's weight vector with itself
            w_norm = (self.weight * self.weight).sum(dim=1) #shape (out_dim)
            w_norm = w_norm.unsqueeze(0)
            factor = dy / w_norm #shape (batch_size, out_dim)
            dx = torch.matmul(factor, self.weight) / self.weight.shape[0] #shape (batch_size, in_dim)
            x_prime = self.last_x + dx

            return x_prime