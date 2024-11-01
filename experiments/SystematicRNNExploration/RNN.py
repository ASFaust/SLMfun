from torch import nn
import torch
from torch.nn import functional as F
from RNNLayer import RNNLayer

class RNN(nn.Module):
    def __init__(self, n_layers, state_size, batch_size,
                 use_input_gate=True, use_forget_gate=False,
                 use_output_gate=False, use_state_gate=False,
                 use_new_state=True, use_old_state=True,
                 use_residual=False):
        """
        Initialize the RNN with input/output embeddings and stacked RNN layers.

        Parameters:
        -----------
        n_layers : int
            The number of RNN layers in the stack.
        state_size : int
            The size of the hidden state for each layer.
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
        super(RNN, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.n_layers = n_layers
        self.batch_size = batch_size

        # Input and output embedding layers for byte-level encoding
        self.input_embedding = nn.Linear(256, state_size, device=self.device)
        self.output_embedding = nn.Linear(state_size, 256, device=self.device)

        # Define a list of RNNLayer layers
        self.layers = nn.ModuleList([
            RNNLayer(input_size=state_size, state_size=state_size, batch_size=batch_size,
                     use_input_gate=use_input_gate, use_forget_gate=use_forget_gate,
                     use_output_gate=use_output_gate, use_state_gate=use_state_gate,
                     use_new_state=use_new_state, use_old_state=use_old_state,
                     use_residual=use_residual)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        """
        Forward pass through the byte-level language model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 256)

        Returns:
        --------
        output : torch.Tensor
            The output tensor with shape (batch_size, 256),
        """
        # Apply input embedding layer
        x = self.input_embedding(x)

        # Apply RNN layers
        for layer in self.layers:
            x = layer(x)

        # Apply output embedding layer to produce logits
        output = self.output_embedding(x)  # (batch_size, sequence_length, 256)

        return output

    def reset(self):
        """
        Reset the hidden states of all RNN layers to zeros.
        """
        for layer in self.layers:
            layer.reset()
