from encoding import encode_string, decode_string
import torch
import numpy as np


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
        '../../../datasets/enwik8.txt',
        batch_size=256,
        history_size=2000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )