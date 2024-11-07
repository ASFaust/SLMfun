import torch
import torch.nn as nn
import math

from DenseNetwork import DenseNetwork

class Net(nn.Module):
    def __init__(self,
                 batch_size,
                 vocab_size,
                 memory_size,
                 memory_dim,
                 hidden_dims,
                 device='cpu',
                 use_tanh=False):
        super(Net, self).__init__()
        # Assign parameters
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dims
        self.use_tanh = use_tanh
        self.device = device

        self.memory = torch.zeros((self.batch_size, self.memory_size, self.memory_dim + self.vocab_size), device=self.device)
        self.memory_mlp = DenseNetwork(self.memory_dim + self.vocab_size, hidden_dims, self.memory_dim, "gated", False, device)

        self.pred_mlp = DenseNetwork(self.memory_dim * self.memory_size, hidden_dims, self.vocab_size, "gated", True, device)

        self.last_state = torch.zeros((self.batch_size, self.memory_dim), device=self.device)

        if self.use_tanh:
            self.tanh = nn.Tanh()

    def forward(self, x):
        # first we update the memory
        with torch.no_grad():
            # first we roll the memory
            self.memory = torch.roll(self.memory, 1, dims=1)
            # then we replace the oldest memory with last_state,x
            self.memory[:, 0, :self.memory_dim] = self.last_state
            self.memory[:, 0, self.memory_dim:] = x.detach()

        # after updating the memory, we compute memory_mlp on each memory slot
        # but since memory has shape (batch_size, memory_size, memory_dim + vocab_size)
        # we need to reshape it to (batch_size * memory_size, memory_dim + vocab_size)
        computed_memory = self.memory_mlp(self.memory.view(-1, self.memory_dim + self.vocab_size))
        if self.use_tanh:
            computed_memory = self.tanh(computed_memory) #to clamp the values between -1 and 1

        # reshape computed_memory back to (batch_size, memory_size + memory_dim)
        computed_memory = computed_memory.view(self.batch_size, -1)
        self.last_state = computed_memory[:, 0:self.memory_dim]
        # finally we compute the prediction
        pred = self.pred_mlp(computed_memory)
        return pred

    def reset(self):
        self.memory = torch.zeros((self.batch_size, self.memory_size, self.memory_dim + self.vocab_size), device=self.device)
        self.last_state = torch.zeros((self.batch_size, self.memory_dim), device=self.device)


    def save(self, path):
        save_dict = {
            "batch_size": self.batch_size,
            "vocab_size": self.vocab_size,
            "memory_size": self.memory_size,
            "memory_dim": self.memory_dim,
            "hidden_dims": self.hidden_dim,
            "use_tanh": self.use_tanh,
            "memory_mlp": self.memory_mlp.state_dict(),
            "pred_mlp": self.pred_mlp.state_dict(),
        }
        torch.save(save_dict, path)

    @staticmethod
    def load(path, use_tanh = False, batch_size=None, device=None):
        load_dict = torch.load(path)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if "vocab_size" in load_dict: #custom as above.
            if not "use_tanh" in load_dict:
                print("use_tanh not found in the model, defaulting to False. If model output seems garbage, try loading with use_tanh=True")
                load_dict["use_tanh"] = use_tanh

            net = Net(load_dict["batch_size"] if batch_size is None else batch_size,
                      load_dict["vocab_size"],
                      load_dict["memory_size"],
                      load_dict["memory_dim"],
                      load_dict["hidden_dims"],
                      use_tanh=load_dict["use_tanh"],
                      device=device)
            net.memory_mlp.load_state_dict(load_dict["memory_mlp"])
            net.pred_mlp.load_state_dict(load_dict["pred_mlp"])
            return net
        else:
            #torch.save(net.state_dict(), path) was used to save the model
            model = torch.load(path) #does this work?
