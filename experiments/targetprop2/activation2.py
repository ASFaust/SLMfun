import torch

class TanhTransformLayer:
    def __init__(self):
        self.running_mean = None
        self.last_x = None
        self.last_y = None

    def forward(self, x):
        self.last_x = x
        if self.running_mean is None:
            self.running_mean = x.mean(dim=0)
        else:
            self.running_mean = 0.9 * self.running_mean + 0.1 * x.mean(dim=0)
        x -= self.running_mean.unsqueeze(0)
        y = (x > 0).float()
        self.last_y = y
        return y

    def backward(self, y_prime):
        target = self.last_x
        #only update the target for the neurons that changed between last_y and y_prime
        target = torch.where(self.last_y < y_prime, 1, target)
        target = torch.where(self.last_y > y_prime, 0, target)
        return target