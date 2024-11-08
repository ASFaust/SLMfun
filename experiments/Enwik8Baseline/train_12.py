from Net import Net
import torch
import time
import json
import os
import numpy as np


def encode_string(to_encode, device):
    """
    Encodes a unicode string into a one-hot tensor of shape [l, 256] with float32 type.
    :param to_encode: bytes to encode
    :param device: Device to store the tensor on
    """
    # Encode the string to bytes and convert directly to a PyTorch tensor
    encoded_bytes = torch.tensor(list(to_encode), dtype=torch.long, device=device)

    # Create a one-hot encoded tensor in float32
    one_hot_encoded = torch.zeros((encoded_bytes.size(0), 256), dtype=torch.float32, device=device)
    one_hot_encoded.scatter_(1, encoded_bytes.unsqueeze(1), 1.0)

    return one_hot_encoded

class DataGenerator:
    def __init__(self, filename, batch_size, history_size, device):
        self.batch_size = batch_size
        self.history_size = history_size
        self.filename = filename
        self.text = ''
        self.text_size = 0
        with open(filename, 'rb') as f:
            self.text = f.read()
        self.text_size = len(self.text)
        self.text_indices = np.frombuffer(self.text, dtype=np.uint8)
        self.device = device

    def get_batch(self):
        start_indices = np.random.randint(0, self.text_size-self.history_size-1, self.batch_size)
        xs = torch.zeros((self.history_size,self.batch_size, 256), device=self.device)
        targets = torch.zeros((self.history_size,self.batch_size), dtype=torch.long, device=self.device)
        for i in range(self.batch_size):
            xs[:,i] = encode_string(self.text[start_indices[i]:start_indices[i]+self.history_size], self.device)
            targets[:,i] = torch.tensor(self.text_indices[start_indices[i]+1:start_indices[i]+1+self.history_size], dtype=torch.long, device=self.device)
        return xs, targets


if __name__ == '__main__':
    generator = DataGenerator(
        '../../datasets/enwik8.txt',
        batch_size=256,
        history_size=2000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

history_size = 2000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1024
training_steps = 500000
log_frequency = 100 #how often to save the loss to the json file

generator = DataGenerator(
    '../../datasets/enwik8.txt',
    batch_size=batch_size,
    history_size=history_size,
    device=device
)

net = Net(
    batch_size=batch_size,
    vocab_size=256,
    memory_size=12,
    hidden_dims=[1024,1024],
    device=device
)

lr = 0.0002

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)

i = 0
ma_loss = 3.0
training_curve = []
smoothed_training_curve = []

while i < training_steps:
    x, targets = generator.get_batch()
    do_break = False
    for j in range(x.shape[1] - 1):
        i += 1
        pred = net.forward(x[j])
        loss = loss_fn(pred, targets[j])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ma_loss = 0.999 * ma_loss + 0.001 * loss.item()
        if i % log_frequency == 0:
            training_curve.append(loss.item())
            smoothed_training_curve.append(ma_loss)
            print(f"Step {i}, Loss {ma_loss}")
            if ma_loss > 10.0:
                do_break = True
                break
    if do_break:
        break

net.save('model_12.pt')
