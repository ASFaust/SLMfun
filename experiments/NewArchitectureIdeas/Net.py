import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,
                 batch_size,
                 n_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 device,
                 early_forget_gate,
                 one_gate_for_all,
                 save_updated_state0,
                 forget_gate_for_state1
        ):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.l_embed = nn.Linear(self.input_size, self.hidden_size, device=self.device)
        self.layers = nn.ModuleList([ResCorrLayer(self.batch_size,
                                                  self.hidden_size,
                                                  self.hidden_size,
                                                  self.hidden_size,
                                                  self.device,
                                                  early_forget_gate,
                                                  one_gate_for_all,
                                                  save_updated_state0,
                                                  forget_gate_for_state1
                                                  ) for _ in range(self.n_layers)])
        self.l_out = nn.Linear(self.hidden_size, self.output_size, device=self.device)

    def forward(self, x):
        x = self.l_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.l_out(x)
        return x

    def reset(self):
        for layer in self.layers:
            layer.reset()

class ResCorrLayer(nn.Module):
    def __init__(self,
                 batch_size,
                 input_size,
                 hidden_size,
                 output_size,
                 device,
                 early_forget_gate,
                 one_gate_for_all,
                 save_updated_state0,
                 forget_gate_for_state1
        ):
        super(ResCorrLayer, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.state0 = torch.zeros((self.batch_size, self.hidden_size), device=self.device)
        self.state1 = torch.zeros((self.batch_size, self.hidden_size), device=self.device)

        state1_gate_output_size = 1 if one_gate_for_all else self.hidden_size
        self.l_0 = nn.Linear(self.input_size + 2 * self.hidden_size, self.hidden_size, device=self.device)
        self.l_1 = nn.Linear(self.input_size + 2 * self.hidden_size, self.hidden_size, device=self.device)
        self.l_2 = nn.Linear(self.input_size + 2 * self.hidden_size, self.hidden_size, device=self.device)
        self.l_3 = nn.Linear(self.input_size + 2 * self.hidden_size, state1_gate_output_size, device=self.device)
        self.l_4 = nn.Linear(self.input_size + 2 * self.hidden_size, self.hidden_size, device=self.device)
        if forget_gate_for_state1:
            self.l_5 = nn.Linear(self.input_size + 2 * self.hidden_size, self.hidden_size, device=self.device)
        self.l_6 = nn.Linear(self.hidden_size + self.input_size, self.output_size, device=self.device)
        self.early_forget_gate = early_forget_gate
        self.one_gate_for_all = one_gate_for_all
        self.save_updated_state0 = save_updated_state0
        self.forget_gate_for_state1 = forget_gate_for_state1

    def forward(self, x):
        concat_input = torch.cat((x, self.state0, self.state1), 1)
        new_state0 = F.tanh(self.l_0(concat_input))
        gate0 = F.sigmoid(self.l_1(concat_input))
        new_state0 = gate0 * new_state0 + (1 - gate0) * self.state0

        if self.early_forget_gate:
            concat_input = torch.cat((x, new_state0, self.state1), 1)
            forget_gate = F.sigmoid(self.l_4(concat_input))
            new_state0 = forget_gate * new_state0

        if not self.save_updated_state0:
            self.state0 = new_state0.detach()

        concat_input = torch.cat((x, new_state0, self.state1), 1)
        new_state1 = F.tanh(self.l_2(concat_input))
        gate1 = F.sigmoid(self.l_3(concat_input))
        if self.one_gate_for_all:
            gate1 = gate1.expand(-1, self.hidden_size)
        new_state1 = gate1 * new_state1 + (1 - gate1) * self.state1

        if self.forget_gate_for_state1:
            concat_input = torch.cat((x, new_state0, new_state1), 1)
            forget_gate = F.sigmoid(self.l_5(concat_input))
            new_state1 = forget_gate * new_state1

        new_state0 += new_state1 - self.state1

        if not self.early_forget_gate:
            concat_input = torch.cat((x, new_state0, self.state1), 1)
            forget_gate = F.sigmoid(self.l_4(concat_input))
            new_state0 = forget_gate * new_state0

        if self.save_updated_state0:
            #this could lead to exploding state values, as it is not bound by tanh...
            self.state0 = new_state0.detach()

        self.state1 = new_state1.detach()

        output = self.l_6(torch.cat((new_state0, x), 1))

        output = output * F.sigmoid(output)

        return output


    def reset(self):
        self.state0 = torch.zeros((self.batch_size, self.hidden_size), device=self.device)
        self.state1 = torch.zeros((self.batch_size, self.hidden_size), device=self.device)
