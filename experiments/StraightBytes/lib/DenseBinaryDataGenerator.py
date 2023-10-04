import torch
import numpy as np

class DenseBinaryDataGenerator:
    def __init__(self, filename, batch_size, bytes_per_step, device):
        self.batch_size = batch_size
        self.filename = filename
        self.text = ''
        self.text_size = 0
        self.bytes_per_step = bytes_per_step
        self.width = bytes_per_step * 8
        with open(filename, 'r') as f:
            self.text = f.read()
        self.text_size = len(self.text)
        self.device = device

    def get_batch(self):
        #should yield a tuple of shape ([batch_size, self.width], [batch_size, 8])
        #the first element is the input, the second is the target
        #the input is a torch tensor of 0s and 1s, which are the concatenated binary representations of the bytes
        #the target is a torch tensor of 0s and 1s, which are the binary representations of the next byte

        #choose random starting positions
        start_positions = np.random.randint(0, self.text_size - self.bytes_per_step - 1, self.batch_size)

        #get the text at those positions, including the next byte
        texts = [self.text[start:start + self.bytes_per_step + 1] for start in start_positions]

        x = []
        y = []

        for text in texts:
            byte_arr = bytearray(text.encode('utf-8'))
            byte_values = np.frombuffer(byte_arr, dtype=np.uint8)
            #now convert them to 8-bit binary
            binary_values = np.unpackbits(byte_values[:-1])
            #now convert to a torch tensor
            binary_tensor = torch.from_numpy(binary_values).float().to(self.device)
            x.append(binary_tensor[-self.width:])
            y.append(byte_values[-1])
        x = torch.stack(x)
        y = torch.tensor(y).to(self.device)
        return x,y
