from lib.DenseBinaryDataGenerator import DenseBinaryDataGenerator
from lib.Dense import Dense
import torch
import time
import wandb

wandb.init(project="Dense Byte Input Language Model")

device = 'cuda'
batch_size = 256
bytes_per_step = 128
n_layers = 4

generator = DenseBinaryDataGenerator(
    '../../datasets/bas.txt',
    batch_size=batch_size,
    bytes_per_step=bytes_per_step,
    device=device
)

net = Dense(
    input_size=generator.width,
    output_size=256,
    n_layers=n_layers,
    device=device
)

wandb.watch(net)

#cross entropy loss that expects a one-hot encoded target and logit output
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

i = 0
ma_loss = 3.0
while True:
    x, y = generator.get_batch()
    i += 1
    pred = net.forward(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    wandb.log({
        "loss": loss.item()
    })
    print("\r{}: {:0.5f}".format(i, loss.item()), end='', flush=True)
    optimizer.step()
    if i % 10000 == 0:
        net.save("models/basmodel.pt")
