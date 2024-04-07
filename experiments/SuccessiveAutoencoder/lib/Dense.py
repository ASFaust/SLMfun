import torch

class Dense(torch.nn.Module):
    def __init__(self, input_size, output_size, n_layers, device, output_activation="linear"):
        super(Dense, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        ratio = output_size / input_size

        intermediate_size = input_size
        for i in range(n_layers - 1):
            next_size = int(input_size * (ratio ** ((i+1) / n_layers)))

            self.layers.append(torch.nn.Linear(intermediate_size, next_size).to(device))
            self.gates.append(torch.nn.Linear(intermediate_size, next_size).to(device))
            print(f"layer {i}, input size {intermediate_size}, output size {next_size}")
            intermediate_size = next_size

        self.layers.append(torch.nn.Linear(intermediate_size, output_size).to(device))

        self.sig = torch.nn.Sigmoid()
        self.device = device

        # Output activation
        if output_activation == "linear":
            self.output_act = lambda x: self.layers[-1](x)
        elif output_activation == "sigmoid":
            self.output_act = lambda x: self.sig(self.layers[-1](x))
        elif output_activation == "tanh":
            self.output_act = lambda x: torch.tanh(self.layers[-1](x))
        elif output_activation == "swiglu":
            self.gates.append(torch.nn.Linear(intermediate_size, output_size).to(device))
            self.output_act = lambda x: self.sig(self.gates[-1](x)) * self.layers[-1](x)
        else:
            raise ValueError(f"Unsupported output activation: {output_activation}")

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            gate = self.sig(self.gates[i](x))
            x = self.layers[i](x)
            x = x * gate
        x = self.output_act(x)  # Apply the output activation
        return x
