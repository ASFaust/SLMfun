from DataGenerator import DataGenerator
from Net import Net
import torch
import time
import wandb
import json
import os

history_size = 4000
device = 'cuda'
batch_size = 256
training_steps = 1000000
log_frequency = 50 #how often to save the loss to the json file

generator = DataGenerator(
    '../../../datasets/enwik8.txt',
    batch_size=batch_size,
    history_size=history_size,
    device=device
)
net_params = {
    'batch_size': batch_size,
    'vocab_size': 256,
    'memory_size': 16,
    'hidden_dim': 1024,
    'max_timesteps': history_size,
    'decay_alpha': 0.95
}

net = Net(**net_params, device=device)

wandb.init(project="Timing and Surprise")
wandb.watch(net)

wandb.config.update(net_params)

#cross entropy loss that expects a target index and logit output
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

i = 0
ma_loss = 3.0
while i < training_steps:
    x, _ = generator.get_batch()
    # x is a tensor of shape (batch_size, history_size, 256)
    for j in range(x.shape[1] - 1):
        i += 1
        inputs = x[:,j].to(device)
        targets = x[:,j+1].argmax(dim=1).to(device)
        pred = net.forward(inputs)
        loss = loss_fn(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        ma_loss = 0.95 * ma_loss + 0.05 * loss.item()

        optimizer.step()

        if i % log_frequency == 0:
            print(f"\r{i}: {ma_loss:.3f}", end='', flush=True)
            wandb.log({'loss': ma_loss})
    #before resetting, i want to know the ages of the memory vectors
    #net_memory_timins is of shape (batch_size, memory_size)
    mean_age = torch.mean(net.memory_timings.float())
    wandb.log({'mean_age': mean_age})
    #also save the memory timings as a histogram to wandb
    wandb.log({'memory_timings': wandb.Histogram(net.memory_timings.cpu().numpy())})
    #and the surprises
    mean_surprise = torch.mean(net.memory_surprise)
    wandb.log({'mean_surprise': mean_surprise})
    wandb.log({'memory_surprise': wandb.Histogram(net.memory_surprise.cpu().numpy())})
    net.reset()
