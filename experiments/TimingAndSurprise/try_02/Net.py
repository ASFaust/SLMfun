import torch
import torch.nn as nn
import math

from DenseNetwork import DenseNetwork

class Net(nn.Module):
    def __init__(self,
                 batch_size,
                 vocab_size,
                 memory_size,
                 hidden_dims,
                 max_timesteps,
                 decay_alpha,
                 device='cpu'):
        super(Net, self).__init__()
        # Assign parameters
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.memory_size = memory_size
        self.hidden_dim = hidden_dims
        self.timing_dim = math.ceil(math.log2(max_timesteps))
        print("timing dimension: ", self.timing_dim)
        self.device = device
        self.decay_alpha = decay_alpha  # Decay factor for surprise

        # Initialize memory, timings, and surprise tensors
        self.memory = torch.zeros((self.batch_size, self.memory_size, self.vocab_size), device=self.device)
        self.memory_timings = torch.zeros((self.batch_size, self.memory_size), dtype=torch.long, device=self.device)
        self.memory_surprise = torch.zeros((self.batch_size, self.memory_size), device=self.device)

        # +1 for normalized timing
        # +1 for surprise
        input_size = (self.vocab_size + self.timing_dim + 1 + 1) * self.memory_size

        print(f"input size: {input_size}")

        self.pred_mlp = DenseNetwork(input_size, hidden_dims, vocab_size, "gated", True, device)

        # Last prediction placeholder
        self.last_prediction = torch.ones((self.batch_size, self.vocab_size), device=self.device) / self.vocab_size

    def forward(self, x):
        surprise = self.compute_surprise(x)
        self.update_memory_timings()  # update the timings of the memory vectors
        self.update_memory(x, surprise)  # delete least surprising and oldest memory vectors and add the new memory vector
        pred_logits = self.compute_prediction()  # based on memory
        # compute softmax of pred_logits
        self.last_prediction = nn.functional.softmax(pred_logits, dim=1).detach()
        return pred_logits

    def compute_surprise(self, x):
        with torch.no_grad():
            surprise = torch.sum(torch.abs(x - self.last_prediction), dim=1)
        return surprise

    def update_memory_timings(self):
        with torch.no_grad():
            self.memory_timings += 1
            # Decay the surprise of the memory vectors to facilitate saving newer information
            self.memory_surprise *= self.decay_alpha

    def update_memory(self, x, surprise):
        # x has shape (batch_size, vocab_size)
        # surprise has shape (batch_size,)
        # this finds the least surprising memory vector and replaces it with the new memory vector
        # find the index of the least surprising memory vector, for each batch
        with torch.no_grad():
            least_surprising_idx = torch.argmin(self.memory_surprise, dim=1)  # has shape (batch_size,)
            # replace the least surprising memory vector with the new memory vector
            self.memory[torch.arange(self.batch_size), least_surprising_idx, :] = x
            # reset the timings of the least surprising memory vector
            self.memory_timings[torch.arange(self.batch_size), least_surprising_idx] = 0
            # and save the surprise of the new memory vector
            self.memory_surprise[torch.arange(self.batch_size), least_surprising_idx] = surprise

    def compute_prediction(self):
        # compute the prediction based on the memory and timings
        # later i want this to be a transformer, but for the first test it will be a 2-layer MLP which gets all memory vectors concatenated
        # but we sort them by timing first, to make it a little easier for the MLP
        sorted_timings, sorted_indices = torch.sort(self.memory_timings, dim=1)
        # sort the memory vectors by the timings
        sorted_memory = self.memory[torch.arange(self.batch_size)[:, None], sorted_indices, :]
        # Convert sorted timings to binary representation
        sorted_timings_binary = ((sorted_timings.unsqueeze(-1) // (2 ** torch.arange(self.timing_dim, device=self.device))) % 2).float()
        # flatten and concat the two tensors
        sorted_memory = sorted_memory.view(self.batch_size, -1)
        sorted_timings_binary = sorted_timings_binary.view(self.batch_size, -1)

        sorted_normalized_timings = sorted_timings.float() / (
                    sorted_timings.max(dim=1, keepdim=True)[0].expand(-1, self.memory_size).float() + 1)
        sorted_normalized_timings = sorted_normalized_timings.view(self.batch_size, -1)

        sorted_surprise = self.memory_surprise[torch.arange(self.batch_size)[:, None], sorted_indices]
        sorted_surprise = sorted_surprise.view(self.batch_size, -1)

        # concatenate the tensors
        pred_input = torch.cat((sorted_memory, sorted_timings_binary, sorted_normalized_timings, sorted_surprise), dim=1)
        # compute the prediction
        pred_logits = self.pred_mlp(pred_input)
        return pred_logits

    def reset(self):
        # Reset memory, timings, and surprise tensors to initial states
        with torch.no_grad():
            self.memory = torch.zeros((self.batch_size, self.memory_size, self.vocab_size), device=self.device)
            self.memory_timings = torch.zeros((self.batch_size, self.memory_size), dtype=torch.long, device=self.device)
            self.memory_surprise = torch.zeros((self.batch_size, self.memory_size), device=self.device)
            # Reset last prediction to uniform distribution
            self.last_prediction = torch.ones((self.batch_size, self.vocab_size), device=self.device) / self.vocab_size

# Example usage
if __name__ == "__main__":
    batch_size = 4
    vocab_size = 10
    memory_size = 5
    hidden_dim = 20
    max_timesteps = 10
    decay_alpha = 0.9
    device = 'cpu'
    net = Net(batch_size, vocab_size, memory_size, hidden_dim, max_timesteps, device)
    #x is a one-hot encoded input tensor of shape (batch_size, vocab_size)
    x = torch.zeros((batch_size, vocab_size), device=device)
    x[torch.arange(batch_size), torch.randint(0, vocab_size, (batch_size,))] = 1
    pred_logits = net.forward(x)
    pred = nn.functional.softmax(pred_logits, dim=1)

    print("Memory:")
    print(net.memory)
    print("Memory timings:")
    print(net.memory_timings)
    print("Memory surprise:")
    print(net.memory_surprise)
    print("Prediction:")
    print(pred)

