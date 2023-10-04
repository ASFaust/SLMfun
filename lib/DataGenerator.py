from encoding import encode_string, decode_string
import torch
import numpy as np

"""

with open('datasets/bas.txt', 'r') as f:
    text = f.read()

# take the first 1000 characters as training data
train_data = text[10000:11000]

# encode the training data, using the GPU if available
encoded = encode_string(train_data, "cpu")[:200]

decoded = decode_string(encoded)

print(decoded)test.py

"""

class DataGenerator:
    def __init__(self, filename, batch_size, history_size, device):
        self.batch_size = batch_size
        self.history_size = history_size
        self.filename = filename
        self.text = ''
        self.text_size = 0
        with open(filename, 'r') as f:
            self.text = f.read()
        self.text_size = len(self.text)
        self.device = device

    def get_batch(self):
        #should yield a tuple of shape ([batch_size, history_size, 256], [batch_size, 256])
        #the first element is the input, the second is the target
        #the input is the one-hot encoded history, the target is the one-hot encoded next character
        #the history is the last history_size characters of the text
        #the target is the character that follows the history
        #the history should be shifted one character to the right for the target
        start_indices = np.random.randint(0, self.text_size-self.history_size-1, self.batch_size)
        x = torch.zeros((self.batch_size, self.history_size, 256), device=self.device)
        y = torch.zeros((self.batch_size, 256), device=self.device)
        for i in range(self.batch_size):
            x[i] = encode_string(self.text[start_indices[i]:start_indices[i]+self.history_size], self.device)
            y[i] = encode_string(self.text[start_indices[i]+self.history_size], self.device)
        return x, y