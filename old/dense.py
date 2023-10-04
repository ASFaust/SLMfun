import torch

class Dense(torch.nn.Module):

    def __init__(self, input_size, output_size, n_layers, device):
        super(Dense, self).__init__()
        print("superdense")
        self.layers = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

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

    def forward(self, x):
        for i in range(len(self.layers)):
            v1 = self.layers[i](x)
            gate = self.act(self.gates[i](x))
            x = v1 * gate
        return x
