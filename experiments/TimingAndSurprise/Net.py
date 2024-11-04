import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, batch_size, vocab_size, memory_size, hidden_dim, timing_dim, device='cpu'):
        super(Net, self).__init__()
        # Assign parameters
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.memory_size = memory_size
        self.hidden_dim = hidden_dim
        self.timing_dim = timing_dim
        self.device = device
        self.alpha = 0.95  # Decay factor for surprise

        # Initialize memory, timings, and surprise tensors
        self.memory = torch.zeros((self.batch_size, self.memory_size, self.vocab_size), device=self.device)
        self.memory_timings = torch.zeros((self.batch_size, self.memory_size), dtype=torch.long, device=self.device)  # Use integer representation for timings
        self.memory_surprise = torch.zeros((self.batch_size, self.memory_size), device=self.device)

        # Initialize the prediction MLP (2-layer MLP)
        self.pred = nn.Sequential(
            nn.Linear(self.vocab_size * self.memory_size + self.memory_size * self.timing_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.vocab_size)
        )

        # Last prediction placeholder
        self.last_prediction = torch.ones((self.batch_size, self.vocab_size), device=self.device) / self.vocab_size

    def forward(self, x):
        surprise = self.compute_surprise(x)
        self.update_memory_timings()  # update the timings of the memory vectors
        self.update_memory(x, surprise)  # delete least surprising and oldest memory vectors and add the new memory vector
        pred_logits = self.compute_prediction()  # based on memory
        # compute softmax of pred_logits
        self.last_prediction = nn.functional.softmax(pred_logits, dim=1)
        return pred_logits

    def compute_surprise(self, x):
        # x is tensor of shape (batch_size, vocab_size) and is a one-hot encoding of the input
        # self.last_prediction is tensor of shape (batch_size, vocab_size) and is the softmax of the last prediction
        # we need to compute the cross-entropy between x and self.last_prediction
        # cross-entropy is the negative log-likelihood of the true class
        surprise = -torch.sum(x * torch.log(self.last_prediction + 1e-6), dim=1)
        return surprise

    def update_memory_timings(self):
        # the timings are always relative to the current time step
        self.memory_timings += 1
        # Decay the surprise of the memory vectors to facilitate saving newer information
        self.memory_surprise *= self.alpha

    def update_memory(self, x, surprise):
        # this finds the least surprising memory vector and replaces it with the new memory vector
        # find the index of the least surprising memory vector, for each batch
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
        # concatenate the two tensors
        pred_input = torch.cat((sorted_memory, sorted_timings_binary), dim=1)
        # compute the prediction
        pred_logits = self.pred(pred_input)
        return pred_logits

    def reset(self):
        # Reset memory, timings, and surprise tensors to initial states
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
    timing_dim = 4  # Use 4 bits for timing representation
    device = 'cpu'

    model = Net(batch_size, vocab_size, memory_size, hidden_dim, timing_dim, device)
    input_data = torch.zeros((batch_size, vocab_size), device=device)
    input_data[0, 1] = 1  # Example one-hot input
    output = model(input_data)
    print(output)
g