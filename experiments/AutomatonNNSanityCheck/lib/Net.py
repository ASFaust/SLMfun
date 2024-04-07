import torch
from lib.Dense import Dense
from lib.AutomatonLayer import AutomatonLayer

class Net(torch.nn.Module):
    def __init__(self, batch_size, hidden_size, num_automata, num_states, device='cuda'):
        super(Net, self).__init__()

        self.input_size = 256
        self.output_size = 256
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_states = num_states
        self.num_automata = num_automata
        self.device = device
        self.l_embed = torch.nn.Linear(self.input_size, self.hidden_size).to(device)

        # Using nn.ModuleList to hold an arbitrary number of GatedStateLayers
        self.l_automata = torch.nn.ModuleList([
            AutomatonLayer(
                batch_size=self.batch_size,
                input_size=self.hidden_size,
                output_size=self.hidden_size,
                num_states=self.num_states,
                device=self.device
            )
            for i in range(num_automata)
        ])
        self.l_intermediate = torch.nn.ModuleList([
            torch.nn.Linear(self.hidden_size * 2, self.hidden_size).to(device) #we dont even need activation functions here
        #    #since dense already does sigmoid multiplication at the beginning
            for i in range(num_automata-1)
        ])

        self.l_decode = torch.nn.Linear(self.hidden_size * 2, self.output_size).to(device)
        self.summary()

        self.silu = torch.nn.SiLU()

    def forward(self, x):
        x = self.l_embed(x)
        state = self.l_automata[0](x)
        for i in range(len(self.l_automata) - 1):
            cat_input = torch.cat((x, state), dim=1)
            x = self.l_intermediate[i](cat_input)
            state = self.l_automata[i+1](x)
        x = torch.cat((x, state), dim=1)
        x = self.l_decode(x)
        return x

    #def forward(self,x):
    #    x = self.silu(self.l_embed(x))
    #    for i in range(len(self.l_automata)):
    #        x = self.l_automata[i](x)
    #    x = self.l_decode(x)
    #    return x

    def detach_state(self):
        for l in self.l_automata:
            l.detach_state()

    def reset(self):
        for l in self.l_automata:
            l.reset()

    def summary(self):
        for name, param in self.named_parameters():
            print(name, param.size())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total number of trainable parameters: {trainable_params}')

    def log(self):
        log_dict = {}
        for i in range(len(self.l_automata)):
            #log_dict[f'automaton {i} selection confidence'] = self.l_automata[i].confidence
            log_dict[f'automaton {i} state stddev']  = self.l_automata[i].last_state_stddev
            log_dict[f'automaton {i} last state'] = self.l_automata[i].last_state[0].detach().cpu().numpy()
            #log_dict[f'automaton {i} state ages'] = self.l_automata[i].state_ages.cpu().numpy() #/ self.l_automata[i].age_counter
            #log_dict[f'automaton {i} normalized state ages'] = self.l_automata[i].state_ages.cpu().numpy() / self.l_automata[i].age_counter
        return log_dict

    def aux_loss(self):
        #confidence_loss from all layers
        confidence_loss = 0
        #for i in range(len(self.l_automata)):
        #    confidence_loss += self.l_automata[i].confidence_loss
        return confidence_loss

    def save(self, path):
        """
        Save the Net model.

        Parameters:
            path (str): Path where the model should be saved.
        """
        save_dict = {
            'batch_size': self.batch_size,
            'hidden_size': self.hidden_size,
            'num_automata': self.num_automata,
            'num_states': self.num_states,
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
            num_automata=data['num_automata'],
            num_states=data['num_states'],
            device=device  # Overriding the saved device with the provided one
        )

        # Load model state
        model.load_state_dict(data['model_state_dict'])

        return model.to(device)
