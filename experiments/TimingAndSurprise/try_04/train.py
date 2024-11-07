from DataGenerator import DataGenerator
from Net import Net
import torch
import time
import json
import os
import wandb
"""
batch_size: 507
lr: 0.0009824994055104307
memory_size: 12
memory_dim: 558
hidden_dims: 629
num_layers: 1
use_tanh: False
"""

history_size = 2000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512
training_time = 60 * 30 # 30 minutes of training - we do that later. i want to play games now
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
    'memory_size': 12,
    'memory_dim': 558,
    'hidden_dims': [629],
}

net = Net(**net_params, device=device)

wandb.init(project="TimingAndSurprise")

wandb.watch(net)

wandb.config.update(net_params)

#cross entropy loss that expects a target index and logit output
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.0005) # almost the same as the one in the optuna study

i = 0
ma_loss = 3.0
start_time = time.time()
while time.time() - start_time < training_time:
    x, targets = generator.get_batch()
    do_break = False
    for j in range(x.shape[1] - 1):
        i += 1
        pred = net.forward(x[j])
        loss = loss_fn(pred, targets[j])

        optimizer.zero_grad()
        loss.backward()
        ma_loss = 0.999 * ma_loss + 0.001 * loss.item()

        optimizer.step()

        if i % log_frequency == 0:
            time_left = training_time - (time.time() - start_time)
            print(f"\r{i}: {ma_loss:.3f}, time left: {time_left:.1f}", end="", flush=True)
            wandb.log({"loss": ma_loss})
            wandb.log({"memory": net.memory[0].detach().cpu().numpy()})

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nStep {i}, NaN/Inf loss, breaking")
            do_break = True
            break

        if ma_loss > 1e5:
            print(f"\nStep {i}, Loss too high, breaking")
            do_break = True
            break


        if time.time() - start_time > training_time:
            print(f"\nStep {i}, Time limit reached, breaking")
            do_break = True
            break

    if do_break:
        break

    net.reset()
    net.save("models/net.pt")