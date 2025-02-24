from network import TargetPropNetwork
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Example usage
net = TargetPropNetwork([28*28, 100, 10],use_bias=True)

# load mnist data
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

def create_target(target_index,  output):
    #get the max activation index of the output for each sample in the batch
    #_, predicted = torch.max(output.data, 1)
    #swap the target index with the predicted index
    #ret = output.clone()
    ##ret[torch.arange(ret.size(0)), target_index] = output[torch.arange(ret.size(0)), predicted]
    #ret[torch.arange(ret.size(0)), predicted] = output[torch.arange(ret.size(0)), target_index]
    #instead of doing that, return a one-hot vector with the target index set to 1
    ret = torch.ones_like(output) * 0
    ret[torch.arange(ret.size(0)), target_index] = 1
    return ret

def measure_accuracy(output, target_index):
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == target_index).sum().item()
    accuracy = correct / output.size(0)
    return accuracy

ma_acc = 0.0

it = 0
while True:
    # train the network
    for batch_idx, (data, target_index) in enumerate(train_loader):
        data = data.view(data.size(0), -1)  # Shape: (batch_size, 28*28)
        output = net.forward(data)
        target = create_target(target_index, output)
        net.backward(target)
        accuracy = measure_accuracy(output, target_index)
        ma_acc = 0.99 * ma_acc + 0.01 * accuracy
        print(f"\rStep {batch_idx}, Accuracy {ma_acc}", end='', flush=True)
        it += 1
        #if it > 10:
        #    exit()