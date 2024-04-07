import torch
from lib.Dense import Dense
from lib.AutomatonLayer import AutomatonLayer


class ParallelAutomatonLayer(torch.nn.Module):
    def __init__(self,
                 batch_size,
                 input_size,
                 output_size,
                 num_states,
                 num_automata,
                 device='cuda'):
        super(ParallelAutomatonLayer, self).__init__()

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_states = num_states
        self.num_automata = num_automata
        self.device = device
        self.automata = torch.nn.ModuleList(
            [AutomatonLayer(
                batch_size,
                input_size,
                output_size / num_automata,
                num_states,
                device
            ) for _ in range(num_automata)]
        )