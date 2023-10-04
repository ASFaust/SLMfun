import torch
from lib.Dense import Dense
#from lib.SaveableModel import SavableModel
from lib.StimulusLayer import StimulusLayer

class Net(torch.nn.Module):
    def __init__(self, batch_size, hidden_size, n_layers, device='cuda'):
        #each layer is a dense layer and a stimulus layer. there is a final dense layer at the end.
        super(Net, self).__init__()
        self.input_size = 256
        self.output_size = 256
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.device = device
        self.l1 = Dense(self.input_size, self.hidden_size, 1, device)
        self.l2 = StimulusLayer(self.batch_size, self.hidden_size, self.hidden_size, device)
        self.l3 = Dense(self.hidden_size, self.hidden_size, 1, device)
        self.l4 = StimulusLayer(self.batch_size, self.hidden_size, self.hidden_size, device)
        self.l5 = Dense(self.hidden_size, self.output_size, 2, device)

    def forward(self, x0):
        x1 = self.l1(x0)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        x5 = self.l5(x4)
        return x5

    def detach_state(self):
        self.l2.detach_state()
        self.l4.detach_state()

    def reset(self):
        self.l2.reset()
        self.l4.reset()

    def save(self, path):
        """
        Save the Net model.

        Parameters:
            path (str): Path where the model should be saved.
        """
        save_dict = {
            'batch_size': self.batch_size,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'device': self.device,
            'model_state_dict': self.state_dict()
        }
        torch.save(save_dict, path)

    @staticmethod
    def load(path, device='cuda'):
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
            batch_size=data['batch_size'],
            hidden_size=data['hidden_size'],
            n_layers=data['n_layers'],
            device=device  # Overriding the saved device with the provided one
        )

        # Load model state
        model.load_state_dict(data['model_state_dict'])

        return model
