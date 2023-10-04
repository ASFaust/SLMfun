import torch

class SaveableModel(torch.nn.Module):
    def __init__(self):
        super(SaveableModel, self).__init__()

    def create_save_function(self, save_args):
        save_dict = {
            'model_state_dict': self.state_dict(),

        }
        torch.save(save_dict, path)

    @staticmethod
    def load(path, batch_size=None, device=None):
        data = torch.load(path)

        # Extract the necessary information from the saved data
        if batch_size is None:
            batch_size = data['batch_size']
        state_size = data['state_size']
        n_history = data['n_history']
        if device is None:
            device = data['device']

        if (device == 'cuda') and not (torch.cuda.is_available()):
            print('CUDA not available, using CPU instead.')
            device = 'cpu'

        net = Net(state_size, n_history, batch_size, device)
        net.load_state_dict(data['model_state_dict'])

        # If you want to ensure the model is moved to the appropriate device:
        net.to(device)

        return net