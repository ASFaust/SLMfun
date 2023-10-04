import torch

from dense import Dense

def logh(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

class Net(torch.nn.Module):
    def __init__(self, batch_size, state_size, n_layers, device):
        super(Net, self).__init__()
        self.input_size = 256
        self.state_size = state_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.device = device
        self.encoder = Dense(self.input_size + self.state_size, self.state_size, self.n_layers, self.device)
        self.decoder = Dense(self.state_size, self.input_size + self.state_size, self.n_layers, self.device)
        self.state = torch.zeros((self.batch_size, self.state_size), device=self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.state_act = torch.nn.Tanh()

    def forward(self, x):
        concat = torch.cat((x, self.state), dim=1)
        encoded = self.state_act(self.encoder(concat))
        all_out = self.decoder(encoded)
        self.reconstructed_input = all_out[:, :self.input_size]
        #softmax
        self.reconstructed_state = self.state_act(all_out[:, self.input_size:])
        state_loss = torch.mean((self.state - self.reconstructed_state) ** 2)
        input_loss = self.loss_fn(self.reconstructed_input, x.argmax(dim=1))
        self.state = encoded
        return state_loss, input_loss

    def detach_state(self):
        self.state = self.state.detach()

    def reset(self):
        self.state = torch.zeros((self.batch_size, self.state_size), device=self.device)

    def save(self, path):
        save_dict = {
            'model_state_dict': self.state_dict(),
            'batch_size': self.batch_size,
            'state_size': self.state_size,
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
        state_size = data['state_size']
        n_layers = data['n_layers']
        if device is None:
            device = data['device']

        if (device == 'cuda') and not (torch.cuda.is_available()):
            print('CUDA not available, using CPU instead.')
            device = 'cpu'

        net = Net(batch_size, state_size, n_layers, device)
        net.load_state_dict(data['model_state_dict'])

        # If you want to ensure the model is moved to the appropriate device:
        net.to(device)

        return net
