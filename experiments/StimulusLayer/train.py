import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))

from DataGenerator import DataGenerator
from model import Net
import torch
import time
import wandb

wandb.init(project="Stimulus Language Model")

history_size = 1000
device = 'cuda'
batch_size = 256
state_size = 256

generator = DataGenerator('datasets/bas.txt', batch_size=batch_size, history_size=history_size, device=device)

net = Net(state_size=state_size, batch_size=batch_size, device=device)

wandb.watch(net)

#cross entropy loss that expects a one-hot encoded target and logit output
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

wandb.config.update({"state_size": state_size, "batch_size": batch_size, "history_size": history_size})

ma_loss = 3.0
i = 0
while True:
    #t0 = time.time()
    i += 1
    x, _ = generator.get_batch()
    pred = net.forward(x)


    wandb.log({"loss": ma_loss, "Top1 error": 1.0 - ma_perc})
    optimizer.zero_grad()
    loss.backward()
    #clip gradients
    torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
    optimizer.step()
    t2 = time.time()
    #print("  {} {}".format(t1-t0, t2-t1), end='\n', flush=True)
    if (i % 1000 == 0):
        net.save("models/model_{:05d}.pt".format(i))
        #print("resetting state")
