from lib.DataGenerator import DataGenerator
from lib.Net import Net
import torch
import time
import wandb


wandb.init(project="Language Model")

history_size = 1000
device = 'cuda'
batch_size = 256
hidden_size = 512
eval_every = 2 #number of bptt unrolls
state_snapshot_every = 100
num_layers = 2

generator = DataGenerator('../../datasets/bas.txt', batch_size=batch_size, history_size=history_size, device=device)

net = Net(batch_size=batch_size, hidden_size=hidden_size, num_layers=num_layers, device=device)
wandb.watch(net)

#cross entropy loss that expects a one-hot encoded target and logit output
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

wandb.config.update({"hidden_size": hidden_size, "batch_size": batch_size, "history_size": history_size})

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
            ma_loss = 0.99 * ma_loss + 0.01 * loss_val
            wandb.log({
                "loss": loss_val
            })
            print("\r{}: {:0.5f}".format(i, ma_loss), end='', flush=True)
            #clip gradients
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
            optimizer.step()
            net.detach_state()
            sum_loss = 0
        if (i % 1000 == 0):
            net.save("models/basmodel.pt".format(i))
            #print("resetting state")
        if i == 50000:
            print("\nadjusting learning rate")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        if i == 100000:
            print("\nadjusting learning rate")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001
        if (i % state_snapshot_every) == 0:
            wandb.log(
                net.log_states()
            )
    net.detach_state()
    sum_loss = 0
    net.reset()