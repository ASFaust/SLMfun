import torch
from torch import nn
from torch.nn import functional as F

class TokenMemoryMachine(nn.Module):
    def __init__(self,
                 batch_size: int,
                 state_size: int,
                 max_memories: int,
                 n_concepts: int,

                 device
            ):
        self.device = device
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.batch_size = batch_size
        self.concepts = [] #shared between batches actually
        self.memories = [] #not shared between batches
        self.concept_usages = []
        self.memory_usages = []
        self.state = torch.zeros((batch_size, token_dim), device=device) # this is just a local recurrent state to keep track of the immediate context


    def forward(self,x):
        #the new memory token gets computed directly from cat[x.detach(),self.state.detach()] and then saved to the memory store
        #the new state is computed with the insights from the systematic rnn exploration experiment
        #and any changes to the state are then applied in a residual manner
        #the changes are compute from the memory store and the concept store.
        #the concept store is directly trainable, and the memory store only computes anew whatever is needed
        pass

    def manage_memory(self):
        pass

