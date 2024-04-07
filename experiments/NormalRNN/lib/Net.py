import torch
from lib.Dense import Dense
from lib.GatedStateLayer import GatedStateLayer
import torch
from lib.Dense import Dense
from lib.GatedStateLayer import GatedStateLayer

class Net(torch.nn.Module):
    def __init__(self, batch_size, hidden_size, num_layers, device='cuda'):
        super(Net, self).__init__()

        self.input_size = 256
        self.output_size = 256
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        self.l_embed = torch.nn.Linear(self.input_size, self.hidden_size).to(device)

        # Using nn.ModuleList to hold an arbitrary number of GatedStateLayers
        self.l_rnns = torch.nn.ModuleList([
            GatedStateLayer(
                batch_size=self.batch_size,
                input_size=self.hidden_size,
                output_size=self.hidden_size,
                state_size=self.hidden_size,
                device=self.device
            )
            for i in range(num_layers)
        ])

        self.l_decode = Dense(self.hidden_size, self.output_size, 2, device)
        self.tanh = torch.nn.Tanh()
        self.sig = torch.nn.Sigmoid()
        self.summary()

    def forward(self, x0):
        x = self.l_embed(x0)

        # Looping through the GatedStateLayers
        for l_rnn in self.l_rnns:
            x = l_rnn(x)

        x = self.l_decode(x)
        return x

    def detach_state(self):
        # Looping through the GatedStateLayers
        for l_rnn in self.l_rnns:
            l_rnn.detach_state()

    def reset(self):
        # Looping through the GatedStateLayers
        for l_rnn in self.l_rnns:
            l_rnn.reset()

    def summary(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model summary:\n\tTrainable parameters: {trainable_params}")

    def log_states(self):
        """
        Construct a dictionary containing the states and their differences for logging.

        Returns:
            dict: A dictionary containing the states and state differences.
        """
        log_dict = {}
        for i, l_rnn in enumerate(self.l_rnns):
            state_key = f"state{i + 1}"
            state_diff_key = f"state{i + 1} diff"

            # Ensure the state and last_state tensors are moved to CPU and detached from computation graph
            state_cpu = l_rnn.state[0].detach().cpu().numpy()
            last_state_cpu = l_rnn.last_state[0].detach().cpu().numpy()

            # Construct dictionary entries
            log_dict[state_key] = state_cpu
            log_dict[state_diff_key] = state_cpu - last_state_cpu

        return log_dict

    def save(self, path):
        """
        Save the Net model.

        Parameters:
            path (str): Path where the model should be saved.
        """
        save_dict = {
            'batch_size': self.batch_size,
            'hidden_size': self.hidden_size,
            'num_layers': len(self.l_rnns),  # Saving the number of layers
            'model_state_dict': self.state_dict()
        }
        torch.save(save_dict, path)

    @staticmethod
    def load(path, device='cuda', batch_size=1):
        """
        Load the Net model.

        Parameters:
            path (str): Path from where the model should be loaded.
            device (str): Device to which the model should be moved.

        Returns:
            Net: Loaded instance of the Net model.
        """
        # Check for CUDA availability and update device if not available
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA is not available, loading model to CPU.")
            device = 'cpu'

        data = torch.load(path, map_location=device)

        # Initialize model
        model = Net(
            batch_size=batch_size,
            hidden_size=data['hidden_size'],
            num_layers=data['num_layers'],  # Utilizing the saved number of layers
            device=device  # Overriding the saved device with the provided one
        )

        # Load model state
        model.load_state_dict(data['model_state_dict'])

        return model.to(device)
