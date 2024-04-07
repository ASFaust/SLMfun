import torch
from lib.Dense import Dense

class GatedStateLayer(torch.nn.Module):
    def __init__(self, batch_size, input_size, output_size, state_size, device='cuda'):
        #each layer is a dense layer and a stimulus layer. there is a final dense layer at the end.
        super(GatedStateLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.state_size = state_size

        self.batch_size = batch_size
        self.device = device

        self.l_state = Dense(self.input_size + self.state_size, self.state_size + self.output_size, 2, device)

        self.state = torch.zeros(self.batch_size, self.state_size).to(device)

        self.tanh = torch.nn.Tanh()

    def forward(self, x0):
        vector = self.l_state(torch.cat((x0, self.state), 1))
        output = vector[:, self.state_size:]
        self.last_state = self.state.clone().detach()
        self.state = self.tanh(vector[:, :self.state_size])
        return output

    def detach_state(self):
        self.state = self.state.detach()

    def reset(self):
        self.state = torch.zeros(self.batch_size, self.state_size).to(self.device)

