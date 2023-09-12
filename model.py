import torch

# the architecture i want to use is a GRU with a linear layer on top

import torch

from dense import Dense


class Net(torch.nn.Module):

    def __init__(self, state_size, n_history, batch_size, device):
        super(Net, self).__init__()
        self.l1_values = Dense(256 + state_size * n_history, state_size, n_layers=3, device=device)
        #self.l1_gates = Dense(256 + state_size * n_history, state_size, n_layers=3, device=device)
        self.l2_prediction = Dense(state_size * n_history, 256, n_layers=3, device=device)

        self.act_values = torch.nn.Tanh()
        #self.act_gates = torch.nn.Sigmoid()

        self.batch_size = batch_size
        self.state_size = state_size
        self.device = device
        self.state = torch.zeros((batch_size, n_history, state_size), device=device)
        self.n_history = n_history
        self.softmax = torch.nn.Softmax(dim=1)

    def update_state(self, new_state):
        #state is a tensor of shape [batch_size, history_size, 256]
        #we roll the state to the left and add the new state to the right
        self.state = torch.cat((self.state[:, 1:, :], new_state.unsqueeze(1)), dim=1)

    def forward_unrolled(self, x):
        # computes its own loss directly. this is for training.
        self.reset()
        # x is a tensor of shape [batch_size, history_size, 256]
        loss_sum = 0.0
        perc_sum = 0.0
        for i in range(x.shape[1] - 1):
            # get the next character
            next_char = x[:, i, :]
            # concatenate the next character with the state
            concat = torch.cat((next_char, self.state.view(self.batch_size, -1)), dim=1)
            # calculate the values and gates
            values = self.act_values(self.l1_values(concat))

            self.update_state(values)

            label = x[:, i+1, :]
            # calculate the loss
            prediction_logits = self.l2_prediction(self.state.view(self.batch_size, -1))
            loss = torch.nn.functional.cross_entropy(prediction_logits, torch.argmax(label, dim=1))
            loss_sum += loss

            #     perc = torch.sum(torch.argmax(a, dim=1) == torch.argmax(y, dim=1)).item() / batch_size
            perc = torch.sum(torch.argmax(prediction_logits, dim=1) == torch.argmax(label, dim=1)).item() / self.batch_size
            perc_sum += perc
        # at the end make a prediction with the state
        return loss_sum, (perc_sum / (x.shape[1] - 1))

    def reset(self):
        self.state = torch.zeros((self.batch_size, self.n_history, self.state_size), device=self.device)

    def forward(self, x):
        with torch.no_grad():
            #x is a tensor of shape [batch_size, 256]
            values = self.act_values(self.l1_values(torch.cat((x, self.state.view(self.batch_size, -1)), dim=1)))
            self.update_state(values)
            return self.l2_prediction(self.state.view(self.batch_size, -1))

    def save(self, path):
        save_dict = {
            'model_state_dict': self.state_dict(),
            'batch_size': self.batch_size,
            'state_size': self.state_size,
            'n_history': self.n_history,
            'device': self.device
        }
        torch.save(save_dict, path)

    @staticmethod
    def load(path, batch_size=None, device=None):
        data = torch.load(path)

        # Extract the necessary information from the saved data
        if batch_size is None:
            batch_size = data['batch_size']
        state_size = data['state_size']
        n_history = data['n_history']
        if device is None:
            device = data['device']

        if (device == 'cuda') and not (torch.cuda.is_available()):
            print('CUDA not available, using CPU instead.')
            device = 'cpu'


        net = Net(state_size, n_history, batch_size, device)
        net.load_state_dict(data['model_state_dict'])

        # If you want to ensure the model is moved to the appropriate device:
        net.to(device)

        return net
