import torch

#we're doing successive autoencoder again yayy
#the principle is the following:
#let m0 be the memory produced as output at time t=0, so the memory input at time t=1
#and i1 the input at time t=1
#and f and g are neural networks
#then we have m1 = f(m0,i1) and m0', i1' = g(m1)
#where m0' and i1' are trained to equal m0 and i1
#this way, m1 is trained to both contain the information of m0 and i1. this makes it a memory of the whole sequence, as it is trained to contain the information of the previous memory and the current input
#to increase its expressiveness, we can add noise to m1 before reconstructing m0' and i1'. this way, we make the memory more robust to noise, which should give us longer-term dependencies
#in practice, this hasn't worked very well YET, but it's a cool idea
#and we can explore it with optuna to find the best hyperparameters

class Net(torch.nn.Module):
    def __init__(self,batch_size,input_size,memory_size,device):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.memory_size = memory_size
        self.device = device
        self.f = Dense(memory_size+input_size,memory_size,2,device,output_activation="linear")
        self.g = Dense(memory_size,memory_size+input_size,2,device,output_activation="linear")
        self.memory = torch.zeros((batch_size,memory_size),device=device)

    def reset(self):
        self.memory = torch.zeros((self.batch_size,self.memory_size),device=self.device)

    def forward(self,x):
        self.memory = self.memory.detach() #detach the memory to avoid backpropagating through the memory in time
        x = torch.cat((x,self.memory),1)
        self.memory = torch.tanh(self.f(x))
        return self.memory

    def reconstruct(self, noise=0):
        x = self.g(self.memory + noise * torch.randn_like(self.memory))
        pred_memory = x[:,:self.memory_size]
        pred_memory = torch.tanh(pred_memory)
        pred_input = x[:,self.memory_size:]
        return pred_memory, pred_input
