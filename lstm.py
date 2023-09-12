import torch

from dense import Dense

def logh(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

class RNNLayer(torch.nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, device):
        super(RNNLayer, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.new_state = Dense(input_size + hidden_size, hidden_size,1, device)
        self.gate = Dense(input_size + hidden_size, hidden_size, 1, device)
        self.output = Dense(input_size + hidden_size, hidden_size, 1, device)

        #self.gate = torch.nn.Linear(input_size + hidden_size, hidden_size).to(device)
        #self.gate2 = torch.nn.Linear(input_size + hidden_size, hidden_size).to(device)
        #self.l_out = Dense(input_size + hidden_size * 2, hidden_size, 2, device)
        #self.new_state = Dense(input_size + hidden_size, hidden_size, 1, device)
        #self.gate = Dense(input_size + hidden_size, hidden_size, 1, device)

        self.act = torch.nn.Tanh()#logh
        self.sigmoid = torch.nn.Sigmoid()

        self.state1 = torch.zeros((batch_size, hidden_size), device=device)
        #self.state2 = torch.zeros((batch_size, hidden_size), device=device)

    def reset(self):
        self.state1 = torch.zeros((self.batch_size, self.hidden_size), device=self.device)
        #self.state2 = torch.zeros((self.batch_size, self.hidden_size), device=self.device)

    def detach_state(self):
        self.state1 = self.state1.detach()
        #self.state2 = self.state2.detach()

    def forward(self, x):
        cat_input = torch.cat((x, self.state1), dim=1)

        new_state1 = self.act(self.new_state(cat_input))

        gate = self.sigmoid(self.gate(cat_input))

        self.state1 = new_state1 * gate + self.state1 * (1 - gate)

        output = self.output(cat_input)

        return output


class Net(torch.nn.Module):
    def __init__(self, hidden_size, n_layers, batch_size, device):
        super(Net, self).__init__()
        self.input_size = 256
        self.output_size = 256
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.device = device
        self.layers = torch.nn.ModuleList()
        if n_layers > 1:
            self.layers.append(RNNLayer(batch_size, self.input_size, hidden_size, device))
            for i in range(max(0,n_layers - 2)):
                self.layers.append(RNNLayer(batch_size, hidden_size, hidden_size, device))
            self.layers.append(RNNLayer(batch_size, hidden_size, self.output_size, device))
        else:
            self.layers.append(RNNLayer(batch_size, self.input_size, self.output_size, device))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def detach_state(self):
        for layer in self.layers:
            layer.detach_state()

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def save(self, path):
        save_dict = {
            'model_state_dict': self.state_dict(),
            'batch_size': self.batch_size,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'device': self.device
        }
        torch.save(save_dict, path)

    @staticmethod
    def load(path, batch_size=None, device=None):
        data = torch.load(path)

        # Extract the necessary information from the saved data
        if batch_size is None:
            batch_size = data['batch_size']
        hidden_size = data['hidden_size']
        n_layers = data['n_layers']
        if device is None:
            device = data['device']

        if (device == 'cuda') and not (torch.cuda.is_available()):
            print('CUDA not available, using CPU instead.')
            device = 'cpu'

        net = Net(hidden_size, n_layers, batch_size, device)
        net.load_state_dict(data['model_state_dict'])

        # If you want to ensure the model is moved to the appropriate device:
        net.to(device)

        return net
