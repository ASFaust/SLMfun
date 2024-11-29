import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
        self.linear1 = TargetPropagationLinear(28*28 + 1, 10, self.bias1)
        self.onehot = TargetPropagationOnehot(self.linear1)

    def forward(self, x):
        x = self.input_hook(x)
        x = self.bias1(x)
        x = self.linear1(x)
        x = self.onehot(x)
        return x

    def backward(self, target):
        """
        Performs backward target propagation.
        :param target: should be a onehot vector of the MNIST label, of shape (batch_size, 10)
        :return: None
        """
        self.onehot.backward(target) #since all others are hooked onto this, we can just call backward on this one.


# Define transformations for the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the MNIST training dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Instantiate your custom network
net = Net()

epochs = 1

ma_acc = 0.0

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        # Flatten the input images into vectors
        data = data.view(data.size(0), -1)  # Shape: (batch_size, 28*28)

        # Forward pass
        output = net.forward(data)

        # Calculate accuracy for monitoring
        _, predicted = torch.max(output.data, 1)


        # Convert target labels to one-hot encoding
        target_onehot = torch.zeros(data.size(0), 10)
        target_onehot.scatter_(1, target.view(-1, 1), 1)

        # Backward pass (updates weights internally)
        net.backward(target_onehot)

        # compute correctly predicted percentage
        correct = (predicted == target).sum().item()
        accuracy = correct / data.size(0)
        ma_acc = 0.999 * ma_acc + 0.001 * accuracy
        print(f"\rEpoch {epoch}, Step {batch_idx}, Accuracy {ma_acc}", end='', flush=True)

#save net.linear1.w as a numpy array to disk
import numpy as np
np.save('linear1_w.npy', net.linear1.w.detach().cpu().numpy())