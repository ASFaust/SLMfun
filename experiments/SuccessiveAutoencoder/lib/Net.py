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
    def __init__(self,
                 batch_size,
                 vocab_size,
                 state_size,
                 device):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.state_size = state_size
        self.device = device
        self.f = torch.nn.Sequential(
            torch.nn.Linear(self.state_size+self.vocab_size,self.state_size),
            torch.nn.Swish(),
            torch.nn.Linear(self.state_size,self.state_size),
            torch.nn.Tanh()
        )

        self.g1 = torch.nn.Sequential(
            torch.nn.Linear(self.state_size,self.state_size),
            torch.nn.Swish(),
            torch.nn.Linear(self.state_size,self.state_size)
        )

        self.g2 = torch.nn.Sequential(
            torch.nn.Linear(self.state_size,self.state_size),
            torch.nn.Swish(),
            torch.nn.Linear(self.state_size + self.vocab_size,self.vocab_size)
        )
        self.memory = torch.zeros((self.batch_size,self.state_size),device=self.device)

    def reset(self):
        self.memory = torch.zeros((self.batch_size,self.state_size),device=self.device)

    def forward_train(self,x):
        new_memory = self.f(torch.cat((self.memory,x),dim=1))
        reconstructed_memory = self.g1(new_memory)
        reconstructed_input = self.g2(torch.cat((new_memory,x),dim=1))
        memory_loss = self.compute_memory_loss(reconstructed_memory,self.memory)
        #x is of shape (batch_size,vocab_size) and is a one-hot encoding of the input
        #so input loss should be the cross-entropy loss between x and reconstructed_input
        input_loss = torch.nn.functional.cross_entropy(reconstructed_input,x.argmax(dim=1))
        self.memory = new_memory.detach() #detach to avoid backpropagating through the memory
        return memory_loss, input_loss

    def compute_memory_loss(self, reconstructed_memory, memory):
        """
        Compute the memory loss between the reconstructed memory and the memory
        The loss is akin to the mean squared error between the memory and the reconstructed memory, but we normalize it by the length of the memory
        to avoid the memory just learning that smaller memory means smaller error
        Args:
            reconstructed_memory:
            memory:

        Returns:

        """
        eps = 1e-6
        diff = memory - reconstructed_memory
        diff = diff ** 2.0
        memory_length_per_batch = torch.sum(memory ** 2.0,dim=1) + eps
        loss = torch.sum(diff,dim=1) / memory_length_per_batch
        return torch.mean(loss)