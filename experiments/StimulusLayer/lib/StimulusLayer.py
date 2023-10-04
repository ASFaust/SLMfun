import torch

class StimulusLayer(torch.nn.Module):
    def __init__(self,batch_size,input_size,output_size,device):
        super(StimulusLayer,self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        #the stimuli. we have as many stimuli as we have as we have output neurons
        #each stimulus is a vector of size input_size
        self.stim = torch.nn.Parameter(torch.randn((output_size,input_size), requires_grad=True, device=device))
        #the scaling and shifting parameters for the stimulus
        self.a = torch.nn.Parameter(torch.randn((output_size,), requires_grad=True, device=device))
        self.b = torch.nn.Parameter(torch.ones((output_size,), requires_grad=True, device=device))
        #self.r = torch.nn.Parameter(torch.ones((output_size,), requires_grad=True, device=device) * 4.0)
        #state holds the time since the last stimulus in an inverse exp decay fashion
        self.l1 = torch.nn.Linear(input_size,output_size,device=device)
        #self.l2 = torch.nn.Linear(input_size,output_size,device=device)
        self.state = torch.zeros((batch_size,output_size), device=device)
        self.device = device

    def forward(self,x):
        #first compute the stimulus
        #the first thing to do is take the norm of x and stim along the input dimension
        self.distances = torch.norm((x - self.stim), p = float('inf'), dim=1)
        #this is of shape (batch_size,output_size)
        value = torch.sigmoid((self.a[None,:] - self.distances) * self.b[None,:])
        #now we have the stimulus, we need to update the state
        #value = torch.sigmoid(self.l1(x))
        self.state = self.state * 0.95 + value
        #now we have the state, we need to update the output
        return self.state

    def reset(self):
        self.state = torch.zeros((self.batch_size,self.output_size), device=self.device)

    def detach_state(self):
        self.state = self.state.detach()
