import torch
import torch.nn as nn

"""
examples of networks that need to be covered:
- regular networks with swish activations
- networks with gates in every layer
- networks that have linear output
- networks that have activated output

-> so we have a flag linear_out, and an activation variable that can either be "swish" or "gated"
"""


class DenseNetwork(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            output_dim,
            activation,
            linear_out,
            device
    ):
        super(DenseNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.linear_out = linear_out
        self.device = device

        self.layers = nn.ModuleList()
        if self.activation == "gated":
            self.gates = nn.ModuleList()
        elif self.activation == "swish":
            self.activation_fn = torch.nn.Swish()
        else:
            raise ValueError("activation must be either 'gated' or 'swish'")

        current_input_dim = input_dim
        for i in range(len(hidden_dims)):
            self.layers.append(nn.Linear(current_input_dim, hidden_dims[i]).to(device))
            if self.activation == "gated":
                self.gates.append(nn.Linear(current_input_dim, hidden_dims[i]).to(device))
            current_input_dim = hidden_dims[i]

        self.layers.append(nn.Linear(current_input_dim, output_dim).to(device))
        if (not self.linear_out) and (self.activation == "gated"):
            self.gates.append(nn.Linear(current_input_dim, output_dim).to(device))

    def forward(self,x):
        for i in range(len(self.layers)):
            next_x = self.layers[i](x)
            if i == (len(self.layers) - 1) and self.linear_out:
                x = next_x
                continue
            if self.activation == "gated":
                next_x = torch.sigmoid(self.gates[i](x)) * next_x
            elif self.activation == "swish":
                next_x = self.activation_fn(next_x)
            else:
                raise ValueError("activation must be either 'gated' or 'swish'")
            x = next_x
        return x
