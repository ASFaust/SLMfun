from network import TargetPropNetwork
import torch

# Example usage
net = TargetPropNetwork(layer_sizes=[4, 8, 16, 4], use_bias=True)
x = torch.randn((4, 4))  # Input with batch_size=2 and input dimension 4

# Forward pass
y = net.forward(x)
print("Output:", y)


# Backward pass (with some dummy error signal y_prime)
y_prime = torch.randn_like(y)
print("Target:", y_prime)

# print the MSE between the target and the output
print("MSE:", torch.nn.functional.mse_loss(y, y_prime))

x_prime = net.backward(y_prime)
print("Backward:", x_prime)

# Then output after the weight update
y = net.forward(x)
print("Output after update:", y)

#print the mse between target before and after the update
print("MSE:", torch.nn.functional.mse_loss(y, y_prime))


