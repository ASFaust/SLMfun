import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net_mnist import Net

# Define transformations for the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the MNIST training dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Instantiate your custom network
net = Net()

epochs = 500

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

