import torch

class ReluLinear:
    def __init__(self, input_size, output_size, device):
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.weights = torch.zeros((self.input_size, self.output_size), device=self.device, requires_grad=False)
        self.v = None
        self.x = None
        self.relu_x = None
        self.relu = torch.nn.ReLU()
        self.alpha = 1.0
        self.dw = None

    def forward(self, x):
        with torch.no_grad():
            self.x = x.clone()  # Shape: (batch_size, input_size)
            self.relu_x = self.relu(x).clone()  # Shape: (batch_size, input_size)
            self.v = torch.matmul(self.relu_x, self.weights)  # Shape: (batch_size, output_size)
        return self.v

    def backward(self, v_hat):
        with torch.no_grad():
            # Compute dv
            dv = v_hat - self.v  # Shape: (batch_size, output_size)

            # Preliminary weight update: dw_pre = alpha * relu_x.T @ dv
            dw_pre = self.alpha * torch.matmul(self.relu_x.t(), dv)  # Shape: (input_size, output_size)

            # Compute preliminary updated weights
            w_hat_pre = self.weights + dw_pre  # Shape: (input_size, output_size)

            # Compute norms
            norm_x_sq = torch.sum(self.relu_x ** 2, dim=1, keepdim=True)  # Shape: (batch_size, 1)
            norm_w_hat_pre_sq = torch.sum(w_hat_pre ** 2)   + 1e-8  # Shape: (1,)

            # Compute the numerator for x_hat
            numerator = dv * (1 - self.alpha * norm_x_sq)  # Shape: (batch_size, output_size)

            # Expand dimensions for broadcasting
            numerator = numerator.unsqueeze(2)  # Shape: (batch_size, output_size, 1)
            w_hat_pre_t = w_hat_pre.t().unsqueeze(0)  # Shape: (1, output_size, input_size)

            # Compute x_hat for each neuron
            x_hat = (numerator / norm_w_hat_pre_sq) * w_hat_pre_t  # Shape: (batch_size, output_size, input_size)

            # Apply ReLU activation and cap values
            x_hat = self.relu(x_hat)
            x_hat = torch.clamp(x_hat, max=2.0)

            # Average x_hat over the output_size dimension to get a common x_hat
            # Mean over relu might underestimate since a lot of values are zero. let's see how it goes
            x_hat_mean = torch.mean(x_hat, dim=1)  # Shape: (batch_size, input_size)

            # Compute <w, x'> and <w, x>
            wx_hat = torch.matmul(x_hat_mean, self.weights)  # Shape: (batch_size, output_size)
            wx = self.v

            # Compute the numerator for dw
            numerator_dw = dv - wx_hat + wx  # Shape: (batch_size, output_size)

            # Compute the denominator |x'|^2
            norm_x_hat_sq = torch.sum(x_hat_mean ** 2, dim=1, keepdim=True) + 1e-8  # Shape: (batch_size, 1)

            # Compute the scaling factor for dw
            scaling_factor = numerator_dw / norm_x_hat_sq  # Shape: (batch_size, output_size)

            # Compute final dw per batch - we don't sum over the batch dimension, we average it
            dw = scaling_factor.unsqueeze(2) * x_hat_mean.unsqueeze(1) # Shape: (batch_size, input_size, output_size)
            self.dw = torch.mean(dw, dim=0)

            # now to backpropagate the gradients to the previous layer through the ReLU activation using x_hat_mean and self.x
            i_prime = torch.zeros_like(x_hat_mean)
            #if x_hat_mean is positive, then we want i_prime to be equal to that positive value
            i_prime[x_hat_mean > 0] = x_hat_mean[x_hat_mean > 0] #this is the part where i think we might be underestimating the target.
            #if x_hat_mean is <= 0, and i_prime < 0, then i_prime = self.x, which indicates no change needed.
            #if x_hat_mean is <= 0, and i_prime >= 0, then i_prime = 0
            i_prime[(x_hat_mean <= 0) & (i_prime < 0)] = self.x[(x_hat_mean <= 0) & (i_prime < 0)]
            i_prime[(x_hat_mean <= 0) & (i_prime >= 0)] = 0

            return i_prime
