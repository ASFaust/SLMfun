import torch

from dense import Dense

class Net(torch.nn.Module):
    def __init__(self, state_size, stack_size, rnn_state_size, batch_size, device):
        super(Net, self).__init__()
        self.rnn_state_size = rnn_state_size

        self.net_new_state_value = Dense(256 + rnn_state_size, state_size, n_layers=3, device=device)
        self.net_new_rnn_state = Dense(256 + rnn_state_size, rnn_state_size, n_layers=3, device=device)

        self.net_context_1 = Dense(rnn_state_size + 256, state_size, n_layers=3, device=device)
        self.net_att_1 = Dense(2*state_size, 1, n_layers=4, device=device)
        self.net_context_2 = Dense(rnn_state_size + state_size, state_size, n_layers=3, device=device)

        self.net_att_2 = Dense(2*state_size, 1, n_layers=4, device=device)
        self.net_prediction = Dense(2 * state_size + rnn_state_size, 256, n_layers=3, device=device)

        self.act_state = torch.nn.Tanh()
        self.act_context = torch.nn.SiLU()
        self.softmax = torch.nn.Softmax(dim=1)

        #self.act_gates = torch.nn.Sigmoid()

        self.batch_size = batch_size
        self.state_size = state_size
        self.stack_size = stack_size

        self.device = device
        self.rnn_state = torch.zeros((self.batch_size, self.rnn_state_size), device=self.device)
        self.state = torch.zeros((self.batch_size, self.stack_size, self.state_size), device=self.device)

    def update_state(self, new_state):
        #state is a tensor of shape [batch_size, stack_size, state_size]
        #we roll the state to the left and add the new state to the right
        self.state = torch.cat((self.state[:, 1:, :], new_state.unsqueeze(1)), dim=1)

    def attention(self, net, context):
        #net is a network that takes in (2*state_size) and outputs a singular value which is a logit
        #context is a tensor of shape [batch_size, state_size]
        #we want to apply the network to each row of the state, along the stack dimension
        #so we need to repeat the context along the stack dimension
        #context is a tensor of shape [batch_size, stack_size, state_size]
        context = context[:, None, :].expand(-1, self.stack_size, -1)
        #context is a tensor of shape [batch_size, stack_size, state_size]
        #and self.state is a tensor of shape [batch_size, stack_size, state_size]
        #now we can concatenate the context and the state
        #net_input is a tensor of shape [batch_size, stack_size, 2*state_size]
        net_input = torch.cat((self.state, context), dim=2)
        #we then apply the network to each row of the state
        #logits is a tensor of shape [batch_size, stack_size]
        net_input = net_input.view(-1, 2*self.state_size)
        logits = net(net_input).view(self.batch_size, self.stack_size)
        #we then apply softmax to get the attention weights
        #weights is a tensor of shape [batch_size, stack_size]
        #maybe another net that scales the logits before softmax based on min, max, mean, std of the logits?
        weights = self.softmax(logits)
        #we then apply the weights to the state to get the result
        #result is a tensor of shape [batch_size, state_size]
        result = torch.sum(self.state * weights[:, :, None], dim=1)
        return result

    def forward(self, x):

        #x is of shape [batch_size, input_size]
        concat_1 = torch.cat([self.rnn_state, x], dim=1)
        #concat has shape [batch_size, rnn_state_size + input_size]
        new_state_value = self.net_new_state_value(concat_1)
        self.rnn_state = self.act_state(self.net_new_rnn_state(concat_1))

        self.update_state(new_state_value) #write the new state to the state stack
        context_1 = self.act_context(self.net_context_1(concat_1))

        val1 = self.attention(self.net_att_1, context_1)
        concat_2 = torch.cat([val1, self.rnn_state], dim=1)
        #concat has shape [batch_size, rnn_state_size + state_size]
        context_2 = self.act_context(self.net_context_2(concat_2))
        val2 = self.attention(self.net_att_2, context_2)
        #now use val1, val2 and self.rnn_state to create the new moving state
        concat_3 = torch.cat([val1, val2, self.rnn_state], dim=1)
        #concat has shape [batch_size, 2*state_size + rnn_state_size]
        #now use the moving state to predict the output, as well as the val1 and val2
        return self.net_prediction(concat_3)

    def detach_state(self):
        self.rnn_state = self.rnn_state.detach()
        self.state = self.state.detach()

    def reset(self):
        self.rnn_state = torch.zeros((self.batch_size, self.rnn_state_size), device=self.device)
        self.state = torch.zeros((self.batch_size, self.stack_size, self.state_size), device=self.device)

    def save(self, path):
        save_dict = {
            'model_state_dict': self.state_dict(),
            'batch_size': self.batch_size,
            'state_size': self.state_size,
            'stack_size': self.stack_size,
            'rnn_state_size': self.rnn_state_size,
            'device': self.device
        }
        torch.save(save_dict, path)

    @staticmethod
    def load(path, batch_size=None, device=None):
        data = torch.load(path)

        # Extract the necessary information from the saved data
        if batch_size is None:
            batch_size = data['batch_size']
        state_size = data['state_size']
        stack_size = data['stack_size']
        rnn_state_size = data['rnn_state_size']
        if device is None:
            device = data['device']

        if (device == 'cuda') and not (torch.cuda.is_available()):
            print('CUDA not available, using CPU instead.')
            device = 'cpu'

        net = Net(state_size, stack_size, rnn_state_size, batch_size, device)
        net.load_state_dict(data['model_state_dict'])

        # If you want to ensure the model is moved to the appropriate device:
        net.to(device)

        return net
