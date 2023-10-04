from lib.DataGenerator import DataGenerator
from lib.Net import Net
import torch
import time
import wandb

wandb.init(project="Stimulus Language Model")

history_size = 1000
device = 'cuda'
batch_size = 256
hidden_size = 256
n_layers = 1
eval_every = 1 #number of bptt unrolls

generator = DataGenerator('../../datasets/nietz.txt', batch_size=batch_size, history_size=history_size, device=device)

net = Net(batch_size=batch_size, hidden_size=hidden_size, n_layers=n_layers, device=device)
wandb.watch(net)

#cross entropy loss that expects a one-hot encoded target and logit output
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

wandb.config.update({"hidden_size": hidden_size, "n_layers": n_layers, "batch_size": batch_size, "history_size": history_size})

i = 0
ma_loss = 3.0
while True:
    x, _ = generator.get_batch()
    sum_loss = 0
    for j in range(x.shape[0] - 1):
        i += 1
        inputs = x[:,j]
        targets = x[:,j+1].argmax(dim=1)
        pred = net.forward(inputs)
        loss = loss_fn(pred, targets)
        sum_loss += loss
        if (i % eval_every == 0):
            optimizer.zero_grad()
            sum_loss.backward()
            loss_val = sum_loss.item() / eval_every
            wandb.log({
                "loss": loss_val,
                "state l2" : net.l2.state[0].detach().cpu().numpy(),
                "state l4" : net.l4.state[0].detach().cpu().numpy()
            })
            print("\r{}: {:0.5f}".format(i, loss_val), end='', flush=True)
            #clip gradients
            #torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
            optimizer.step()
            net.detach_state()
            sum_loss = 0
        if (i % 1000 == 0):
            net.save("models/model2.pt".format(i))
            #print("resetting state")
    net.detach_state()
    sum_loss = 0
    net.reset()