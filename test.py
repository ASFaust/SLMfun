from DataGenerator import DataGenerator
from model import Net
import torch

history_size = 1
device = 'cpu'
batch_size = 1
state_size = 512

generator = DataGenerator('datasets/bas.txt', batch_size=batch_size, history_size=history_size, device=device)

net = Net(state_size=256, batch_size=batch_size, device=device)

#cross entropy loss that expects a one-hot encoded target and logit output
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)

ma_loss = 3.0
for i in range(10000):
    optimizer.zero_grad()
    x, y = generator.get_batch()
    a = net.forward_unrolled(x)
    loss = loss_fn(a, torch.argmax(y, dim=1))
    ma_loss = 0.99*ma_loss + 0.01*loss.item()
    print("\r{}/{}: {}".format(i, 100, ma_loss), end='', flush=True)
    loss.backward()
    optimizer.step()
