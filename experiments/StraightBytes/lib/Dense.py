import torch

class Dense(torch.nn.Module):

    def __init__(self, input_size, output_size, n_layers, device):
        super(Dense, self).__init__()
        print(f"Creating Dense Network: input_size={input_size}, output_size={output_size}, n_layers={n_layers}, device={device}")
        self.layers = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers

        ratio = output_size / input_size

        intermediate_size = input_size
        for i in range(n_layers):
            next_size = int(input_size * (ratio ** ((i+1) / (n_layers))))
            if i == (n_layers - 1):
                next_size = output_size
            print("layer", i, "input size", intermediate_size, "output size", next_size)

            self.layers.append(torch.nn.Linear(intermediate_size, next_size).to(device))
            self.gates.append(torch.nn.Linear(intermediate_size, next_size).to(device))
            intermediate_size = next_size

        self.act = torch.nn.Sigmoid()
        self.device = device

        #count parameters
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        print(f"Total parameters: {total_params}")

    def forward(self, x):
        for i in range(len(self.layers)):
            v1 = self.layers[i](x)
            gate = self.act(self.gates[i](x))
            x = v1 * gate
        return x

    def save(self, path):
        """
        Save the Dense model.

        Parameters:
            path (str): Path where the model should be saved.
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'device': self.device,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'n_layers': self.n_layers
        }
        torch.save(save_dict, path)

    @staticmethod
    def load(path, device='cuda'):
        """
        Load the Dense model.

        Parameters:
            path (str): Path from where the model should be loaded.
        """
        save_dict = torch.load(path, map_location=device)
        model = Dense(save_dict['input_size'], save_dict['output_size'], save_dict['n_layers'], save_dict['device'])
        model.load_state_dict(save_dict['model_state_dict'])
        return model
