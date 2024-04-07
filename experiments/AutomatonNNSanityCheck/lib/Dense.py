import torch

class Dense(torch.nn.Module):

    def __init__(self, input_size, output_size, n_layers, device, sigmoid_output=False):
        super(Dense, self).__init__()
        #print(f"Creating Dense Network: input_size={input_size}, output_size={output_size}, n_layers={n_layers}, device={device}, sigmoid_output={sigmoid_output}")
        self.layers = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        ratio = output_size / input_size

        intermediate_size = input_size
        #first gating layer to focus on the most important features
        self.gates.append(torch.nn.Linear(intermediate_size, intermediate_size).to(device))
        for i in range(n_layers):
            next_size = int(input_size * (ratio ** ((i+1) / (n_layers))))
            if i == (n_layers - 1):
                next_size = output_size
            #print("layer", i, "input size", intermediate_size, "output size", next_size)

            if i == (n_layers - 1):
                self.gates.append(torch.nn.Linear(intermediate_size, next_size).to(device))
                if not sigmoid_output:
                    self.layers.append(torch.nn.Linear(intermediate_size, next_size).to(device))
            else:
                self.layers.append(torch.nn.Linear(intermediate_size, next_size).to(device))
                self.gates.append(torch.nn.Linear(intermediate_size, next_size).to(device))
            intermediate_size = next_size

        self.act = torch.nn.Sigmoid()
        self.device = device
        self.sigmoid_output = sigmoid_output

        #count parameters
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        #print(f"Total parameters: {total_params}")

    def forward(self, x):
        gate = self.act(self.gates[0](x))
        x = x * gate
        for i in range(len(self.gates) - 2):
            v1 = self.layers[i](x)
            gate = self.act(self.gates[i+1](x))
            x = v1 * gate

        sig = self.act(self.gates[-1](x))

        if not self.sigmoid_output:
            x = sig * self.layers[-1](x)
        else:
            x = sig
        return x