import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net_mnist import Net

# Assuming Net is defined as per your code
# from your_module import Net

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

# Number of epochs to train
num_epochs = 1

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Flatten the input images into vectors
        data = data.view(data.size(0), -1)  # Shape: (batch_size, 28*28)

        # Forward pass
        output = net.forward(data)

        # Convert target labels to one-hot encoding
        target_onehot = torch.zeros(data.size(0), 10)
        target_onehot.scatter_(1, target.view(-1, 1), 1)

        # Backward pass (updates weights internally)
        net.backward(target_onehot)

        # Calculate accuracy for monitoring
        _, predicted = torch.max(output.data, 1)

        #now perform forward pass again to see if the target is reached
        output2 = net.forward(data)
        _, predicted2 = torch.max(output2.data, 1)

        print("predicted: ", predicted)
        print("target: ", target)
        print("predicted after training: ", predicted2)
        exit(1)

print("Training complete.")
