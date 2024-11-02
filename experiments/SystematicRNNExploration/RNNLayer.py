import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from graphviz import Digraph
from itertools import product


class RNNLayer(nn.Module):
    def __init__(self, input_size, state_size, batch_size,
                 use_input_gate=False, use_forget_gate=False,
                 use_output_gate=False, use_state_gate=False,
                 use_new_state=True, use_old_state=True,
                 use_residual=False):
        """
        Initialize the RNNLayer.

        Parameters:
        -----------
        input_size : int
            The size of the input features, as well as the output size.
        state_size : int
            The size of the hidden state.
        batch_size : int
            The batch size.
        use_input_gate : bool, optional
            Whether to use an input gate (default is False).
        use_forget_gate : bool, optional
            Whether to use a forget gate (default is False).
        use_output_gate : bool, optional
            Whether to use an output gate (default is False).
        use_state_gate : bool, optional
            Whether to use a state gate (default is False).
        use_new_state : bool, optional
            Whether to include new state in the output (default is True).
        use_old_state : bool, optional
            Whether to include old state in the output (default is True).
        use_residual : bool, optional
            Whether to add a residual connection (default is False).
        """
        super(RNNLayer, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize parameters
        self.input_size = input_size
        self.state_size = state_size
        self.batch_size = batch_size
        self.use_input_gate = use_input_gate
        self.use_forget_gate = use_forget_gate
        self.use_output_gate = use_output_gate
        self.use_state_gate = use_state_gate
        self.use_new_state = use_new_state
        self.use_old_state = use_old_state
        self.use_residual = use_residual

        # Validate output configuration
        if not (self.use_new_state or self.use_old_state):
            raise ValueError("At least one of use_new_state or use_old_state must be True.")

        # Initialize state
        self.state = torch.zeros((self.batch_size, self.state_size), device=self.device)

        # Define the primary state neural network
        self.state_nn = nn.Linear(input_size + state_size, state_size, device=self.device)

        # Calculate the input width for output_nn and output_gate_nn based on settings
        output_input_dim = 0
        if self.use_new_state:
            output_input_dim += state_size
        if self.use_old_state:
            output_input_dim += state_size
        if self.use_residual:
            output_input_dim += input_size

        # Define output neural network with the computed input dimension
        self.output_nn = nn.Linear(output_input_dim, input_size, device=self.device)

        # Define gates if specified
        if self.use_input_gate:
            self.input_gate_nn = nn.Linear(input_size + state_size, input_size, device=self.device)
            self.state_input_gate_nn = nn.Linear(input_size + state_size, state_size, device=self.device)

        if self.use_state_gate:
            self.state_gate_nn = nn.Linear(input_size + state_size + state_size, state_size, device=self.device)

        if self.use_forget_gate:
            self.forget_gate_nn = nn.Linear(input_size + state_size + state_size, state_size, device=self.device)

        if self.use_output_gate:
            self.output_gate_nn = nn.Linear(state_size + output_input_dim, input_size, device=self.device)

    def forward(self, x):
        """
        Forward pass of the RNN layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns:
        --------
        output : torch.Tensor
            The output tensor after processing through the RNN layer.
        """
        old_state_inp = self.state
        x_inp = x
        combined = torch.cat((x_inp, old_state_inp), dim=1)
        if self.use_input_gate:
            x_inp = torch.sigmoid(self.input_gate_nn(combined)) * x
            state_input_gate = torch.sigmoid(self.state_input_gate_nn(combined))
            old_state_inp = state_input_gate * self.state
            combined = torch.cat((x_inp, old_state_inp), dim=1)

        new_state = torch.tanh(self.state_nn(combined))

        if self.use_state_gate:
            gate_input = torch.cat((combined, new_state), dim=1)
            state_gate = torch.sigmoid(self.state_gate_nn(gate_input))
            new_state = state_gate * new_state + (1.0 - state_gate) * self.state

        if self.use_forget_gate:
            gate_input = torch.cat((combined, new_state), dim=1)
            forget_gate = torch.sigmoid(self.forget_gate_nn(gate_input))
            new_state = forget_gate * new_state

        if self.use_new_state and not self.use_old_state:
            input_to_output = new_state
        elif self.use_old_state and not self.use_new_state:
            input_to_output = old_state_inp
        elif self.use_new_state and self.use_old_state:
            input_to_output = torch.cat((new_state, old_state_inp), dim=1)
        else:
            raise ValueError("At least one of use_new_state or use_old_state must be True.")

        if self.use_residual: #not a true residual connection, but information can now flow separately from the input to the output
            input_to_output = torch.cat((input_to_output, x), dim=1)

        output = self.output_nn(input_to_output)

        if self.use_output_gate:
            output_gate = torch.sigmoid(self.output_gate_nn(torch.cat((output, input_to_output), dim=1)))
            output = output * output_gate
        else:
            # use swish activation function, which is basically like a self-gating mechanism
            output = output * torch.sigmoid(output)

        self.state = new_state.detach()

        return output

    def reset(self):
        """
        Reset the hidden state to zeros.
        """
        self.state = torch.zeros((self.batch_size, self.state_size), device=self.device)


"""
separate idea for a RNN structure:

def forward(self, x):
    combined_input = torch.cat((x, self.state), dim=1)
    state_gate = torch.sigmoid(self.state_gate_nn(combined_input))
    gated_state = state_gate * self.state
    input_to_output = torch.cat((x, gated_state), dim=1)
    output = self.output_nn(input_to_output)
    output = output * torch.sigmoid(output)

#this preliminary forward pass focuses on the information in the state
#which is not directly observable from the input, and only on the information that
#is relevant for the current output.
#now we need to decide on how to update the state.
#if we had BPTT, we would just update the state like this:    

def forward(self, x):
    combined_input = torch.cat((x, self.state), dim=1)
    state_gate = torch.sigmoid(self.state_gate_nn(combined_input))
    gated_state = state_gate * self.state
    input_to_output = torch.cat((x, gated_state), dim=1)
    output = self.output_nn(input_to_output)
    output = output * torch.sigmoid(output)
    new_state = torch.tanh(self.state_nn(combined_input))
    self.state = new_state 

there really needs to be a differentiation in the model between:

- adding information to the state
- making predictions conditionally on the state and the input, as well as a timing signal
- writing and reading policies for the state

the state should be a memory that holds information about the state of the world, and not a memory that needs to learn timing and computation.
the computation on facts should be done by a separate network, which updates the sstate
so the state gets update by computations done on the input and the state itself
but the state should not be directly used to compute the output. the output should be computed by a separate network, which 
gets a current context, and the state as input
so there needs to be two states basically
one that defines the immediate context, and one that defines the long term context
the long term context should be updated by the immediate context, and the immediate context should be updated by the input
the output should be computed by the immediate context and the long term context
the immediate context should be updated by the input and the long term context
the long term context should be updated by the immediate context and the long term context
so we have self.state and self.long_term_state

but how do we update the long term state?
we could generate a gradient into the long term state via the immediate state:

def forward(self, x):
    new_long_term_state = self.long_term_state_nn(torch.cat((x, self.state, self.long_term_state), dim=1))
    new_state = self.state_nn(torch.cat((x, self.state, new_long_term_state), dim=1))
    input_to_output = torch.cat((x, new_state, new_long_term_state.detach()), dim=1) #in order to not update the long term state with the gradient from the input
    self.state = new_state.detach()
    self.long_term_state = new_long_term_state.detach()
    

no uh

lets start again, im not satisfied. 

we will use residual things for the first choice:

def forward(self,x):
    out1 = self.l1(x) #the first prediction is just a markov chain
    out1 += self.l2(x,last_out) #the second prediction is the first prediction plus the markov chain
    last_out = out1.detach() #we detach the last output, so that the gradient does not flow back to the first prediction
    return self.swish(out1)

def forward(self, x):
    out1 = self.l1(x)  # The first prediction is just a Markov chain
    state1 = state1 * torch.sigmoid(self.forget_gate(out1, x, state1)) * 0.9  # Decay and forget the state
    state1 = state1 + self.swish(self.l2((out1, x, state1)))  # Add some information to the state
    out1 += self.l2((x, state1))
    return out1

def forward(self, x):
    out1 = self.l1(x)  # The first prediction is just a Markov chain
    state1 = state1 * 0.9  # Decay the state to ensure no infinite growth
    state1_gate = torch.sigmoid(self.state1_view_gate(out1, x, state1))
    state1_view = state1 * state1_gate #view just the important parts of the state
    state1 = state1 * (1.0 - state1_gate) + state1_gate * self.l2((out1, x, state1_view))  # Add some information to the state
    out1 += self.l2(state1_view)
    state1 = state1.detach()  # Detach the state
    return self.swish(out1)


def forward(self, x):
    out1 = self.l1(x)
    new_state1 = self.l2((out1, x, self.state1))
    out1 += new_state1 - self.state1 #this is genius: if no error is predicted by the state, the state is not updated
    new_state2 = self.l3((out1, x, self.state2))
    out1 += new_state2 - self.state2
    self.state1 = new_state1.detach()
    self.state2 = new_state2.detach()
    return self.swish(out1)

"""



