import numpy as np
import torch
from lib.Dense import Dense

"""
the output is a softmax weighted sum of the matrix rows, 
where the softmax is determined by the input and the last state. 
the matrix entries are learned fixed parameters.
"""

class AutomatonLayer(torch.nn.Module):
    def __init__(self, batch_size, input_size, output_size, num_states, device='cuda'):
        super(AutomatonLayer, self).__init__()

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_states = num_states
        self.device = device
        #a matrix of size num_states x output_size
        #self.state_matrix = torch.nn.Parameter(torch.randn(num_states, output_size).to(device))

        #self.l_selection = Dense(self.input_size + self.output_size, self.num_states, 2, device)
        #self.l_selection = torch.nn.Linear(self.input_size + self.output_size, self.output_size).to(device)
        self.l_selection = Dense(self.input_size + self.output_size, self.output_size, 2, device)

        self.last_state = torch.zeros(self.batch_size, self.output_size).to(device)

        #a continous approximation to when last the state was used in a prediction.
        self.state_ages = torch.zeros(self.num_states).to(self.device)

        self.age_counter = 0

        self.softmax = torch.nn.Softmax(dim=1)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x0):
        self.last_state = self.tanh(self.l_selection(torch.cat((x0, self.last_state), 1)))

        #selection has now shape [batch_size, num_states]
        #state matrix has shape [num_states, output_size]
        #self.last_state = torch.einsum('so,bs->bo', self.state_matrix, selection)

        #below are metrics
        self.last_state_stddev = self.last_state.detach().cpu().numpy().std()
        #self.state_ages = ((1.0 - torch.max(selection, dim=0)[0]) * (self.state_ages + 1.0)).detach()
        #self.age_counter += 1.0
        #self.confidence = selection.detach().cpu().numpy().max()
        return self.last_state

    def detach_state(self):
        self.last_state = self.last_state.detach()

    def reset(self):
        self.last_state = torch.zeros(self.batch_size, self.output_size).to(self.device)
        self.state_ages = torch.zeros(self.num_states).to(self.device)
        self.age_counter = 0



