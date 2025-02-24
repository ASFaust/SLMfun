from linear import TargetPropLinear
import torch

# Example usage
net = TargetPropLinear(5, 3)
x = torch.randn((6, 5))  # Input with batch_size=2 and input dimension 4
x[:,:1] = 1.0
# Forward pass
y = net.forward(x)
#print("Output:", y)

# Backward pass (with some dummy error signal y_prime)
y_prime = torch.randn_like(y)

#print("Target:", y_prime)

# print the MSE between the target and the output
print("MSE:", torch.nn.functional.mse_loss(y, y_prime))

for i in range(1):
    a = net.backward(y_prime)
    y = net.forward(x)
    # Then output after the weight update
    #y = net.forward(x)
    #print("Output after update:", y)

    #print the mse between target before and after the update
    print("MSE:", torch.nn.functional.mse_loss(y, y_prime))

    #print("x:", x)
    #print("x_prime:", a)
    y2 = net.forward(a) #this should yield smaller mse
    print("MSE with new input:", torch.nn.functional.mse_loss(y2, y_prime))
    x = a