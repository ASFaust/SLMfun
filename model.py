import torch

# the architecture i want to use is a GRU with a linear layer on top

class Dense(torch.nn.Module):

    def __init__(self, input_size, output_size, n_layers=1, device='cpu'):
        super(Dense, self).__init__()
        self.layers = torch.nn.ModuleList()
        intermediate_size = input_size
        for i in range(n_layers):
            interp = i / (n_layers - 1)
            next_size = int(interp * output_size + (1.0 - interp) * input_size)
            if i == (n_layers - 1):
                next_size = output_size
            self.layers.append(torch.nn.Linear(intermediate_size, next_size))
            # slowly interpolate between input_size and output_size for intermediate sizes
            intermediate_size = next_size
        self.act = torch.nn.Tanh()
        self.device = device

    def forward(self, x):
        # dont activate the last layer
        for i in range(len(self.layers) - 1):
            x = self.act(self.layers[i](x))
        return self.layers[-1](x)


class Net(torch.nn.Module):

    def __init__(self, state_size, batch_size, device):
        super(Net, self).__init__()
        self.l1_values = Dense(256 + state_size, state_size, n_layers=4, device=device)
        self.l1_gates = Dense(256 + state_size, state_size, n_layers=4, device=device)
        self.l2_prediction = Dense(state_size, 256, n_layers=4, device=device)

        self.act_values = torch.nn.Tanh()
        self.act_gates = torch.nn.Sigmoid()

        self.batch_size = batch_size
        self.state_size = state_size
        self.device = device
        self.state = torch.zeros((batch_size, state_size), device=device)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward_unrolled(self, x):
        # returns a tensor of shape [batch_size, 256] with linear activation -> logits!
        self.reset()
        # x is a tensor of shape [batch_size, history_size, 256]
        for i in range(x.shape[1]):
            # get the next character
            next_char = x[:, i, :]
            # concatenate the next character with the state
            concat = torch.cat((next_char, self.state), dim=1)
            # calculate the values and gates
            values = self.act_values(self.l1_values(concat))
            gates = self.act_gates(self.l1_gates(concat))
            # update the state
            self.state = (1 - gates) * self.state + gates * values
        # at the end make a prediction with the state
        return self.l2_prediction(self.state)

    def reset(self):
        self.state = torch.zeros((self.batch_size, self.state_size), device=self.device)

    def forward(self, x):
        #declare no gradients
        # state is a tensor of shape [batch_size, 256]
        # x is a tensor of shape [batch_size, 256], the new character
        # returns a tensor of shape [batch_size, 256] with softmax applied
        # concatenate the next character with the state
        with torch.no_grad():
            concat = torch.cat((x, self.state), dim=1)
            # calculate the values and gates
            values = self.act_values(self.l1_values(concat))
            gates = self.act_gates(self.l1_gates(concat))
            # update the state
            self.state = (1 - gates) * self.state + gates * values
            # at the end make a prediction with the state
            return self.softmax(self.l2_prediction(self.state))
