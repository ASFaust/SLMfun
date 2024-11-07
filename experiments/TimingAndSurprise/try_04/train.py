from DataGenerator import DataGenerator
from Net import Net
import torch
import time
import json
import os
import wandb

history_size = 2000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256
training_steps = 100000
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
    'memory_size': 8,
    'memory_dim': 256,
    'hidden_dims': [1024, 1024]
}

net = Net(**net_params, device=device)

wandb.init(project="TimingAndSurprise")

wandb.watch(net)


wandb.config.update(net_params)

#cross entropy loss that expects a target index and logit output
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.01) #label smoothing to prevent nan loss

optimizer = torch.optim.Adam(net.parameters(), lr=0.002)

i = 0
ma_loss = 3.0
while i < training_steps:
    x, targets = generator.get_batch()
    for j in range(x.shape[1] - 1):
        i += 1
        pred = net.forward(x[j])
        loss = loss_fn(pred, targets[j])

        optimizer.zero_grad()
        loss.backward()
        ma_loss = 0.95 * ma_loss + 0.05 * loss.item()

        optimizer.step()

        if i % log_frequency == 0:
            print(f"\r{i}: {ma_loss:.3f}", end='', flush=True)
            wandb.log({"loss": ma_loss})
            wandb.log({"memory": net.memory[0].detach().cpu().numpy()})
    net.reset()
    net.save("models/net.pt")