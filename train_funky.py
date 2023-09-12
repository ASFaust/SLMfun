from DataGenerator import DataGenerator
from funkymodel import Net
import torch
import time
import wandb

wandb.init(project="SLMfun")

history_size = 5000
device = 'cuda'
batch_size = 256
state_size = 64
stack_size = 128
rnn_state_size = 128

generator = DataGenerator('datasets/bas.txt', batch_size=batch_size, history_size=history_size, device=device)

net = Net(state_size=state_size, stack_size=stack_size, rnn_state_size=rnn_state_size, batch_size=batch_size, device=device)

wandb.watch(net)

#cross entropy loss that expects a one-hot encoded target and logit output
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)

eval_every = 10

#wandb.config.update({"state_size": state_size, "stack_size": stack_size, "batch_size": batch_size, "history_size": history_size})

ma_loss = 3.0
ma_perc = 0.0
i = 0
while True:
    #t0 = time.time()
    x, _ = generator.get_batch()
    sum_loss = 0
    for j in range(x.shape[0] - 1):
        i += 1
        inputs = x[:,j]
        targets = x[:,j+1].argmax(dim=1)
        logits = net.forward(inputs)
        loss = loss_fn(logits, targets)
        sum_loss += loss
        if (i % eval_every == 0):
            optimizer.zero_grad()
            sum_loss.backward()
            loss_val = sum_loss.item() / eval_every
            wandb.log({"loss": loss_val})
            print("\r{}: {:0.5f}".format(i, loss_val), end='', flush=True)
            #clip gradients
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
            optimizer.step()
            net.detach_state()
            sum_loss = 0
        if (i % 1000 == 0):
            net.save("models/model_{:05d}.pt".format(i))
            # print("resetting state")
    net.detach_state()
    sum_loss = 0
    net.reset()
