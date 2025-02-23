import torch


class TargetPropLinear:
    def __init__(self, in_dim, out_dim):
        self.weight = torch.randn((out_dim, in_dim), requires_grad=False)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.last_y = None
        self.last_x = None

    def forward(self, x):
        self.last_x = x
        y = torch.matmul(x, self.weight.T)
        self.last_y = y
        return y

    def backward(self, y_prime):
        batch_size = y_prime.shape[0]

        # Sample a direction d
        d = torch.randn(self.in_dim)

        # Compute <x, d>
        xd = torch.matmul(self.last_x, d)  # Shape (batch_size,)
        xd = xd.unsqueeze(1)  # Shape (batch_size, 1)

        # Compute dy
        dy = y_prime - self.last_y  # Shape (batch_size, out_dim)
        targets = dy / xd  # Shape (batch_size, out_dim)
        target = targets.mean(dim=0)  # Shape (out_dim,)

        # Update weight matrix
        dw = torch.outer(target, d)  # Shape (out_dim, in_dim)
        self.weight += dw

        # Compute target of the previous layer using <w, d>
        wd = torch.matmul(self.weight, d)  # Shape (out_dim,)
        wd = wd.unsqueeze(0).repeat(batch_size, 1)  # Shape (batch_size, out_dim)
        y = self.forward(self.last_x)
        dy = y_prime - y
        targets = dy / wd  # Shape (batch_size, out_dim)
        target = targets.mean(dim=1)  # Shape (batch_size,)
        x_prime = self.last_x + d.unsqueeze(0) * target.unsqueeze(1)  # Shape (batch_size, in_dim)

        return x_prime
