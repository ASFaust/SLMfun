import torch
from torch import nn
from torch.nn import functional as F

def double_stack(list_of_lists):
    """
    double stack a list of lists
    :param list_of_lists:
    :return: torch.tensor with shape (batch_size, <=max_tokens, token_dim)
    """


class TokenMemoryMachine(nn.Module):
    def __init__(self,
                 batch_size: int,
                 max_tokens: int,
                 token_dim: int,
                 device
            ):
        self.device = device
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.batch_size = batch_size
        self.token_values = torch.zeros((batch_size, max_tokens, token_dim), device=device)
        self.token_keys = torch.zeros((batch_size, max_tokens, token_dim), device=device)
        self.token_usages = torch.zeros((batch_size, max_tokens), device=device)

    def parse_input(self,x):
        """
        parse a new input x into the token memory machine
        this is done with multi head attention across the token memory machine
        :param x:
        :return:
        """
        output = self.l_embed(x.detach())
        min_indices = torch.argmin(self.token_usages, dim=1)
        self.token_values[range(self.batch_size), min_indices] = output

    def parse_input_query(self,x):
        """
        parse the input x into a query
        :param x:
        :return:
        """
        pass

    def calculate_attention_weights(self, query):
        """
        calculate the attention weights
        :param query:
        :return:
        """
        pass

    def calculate_output(self, attention_weights):
        """
        calculate the output
        :param attention_weights:
        :return:
        """
        pass

    def generate_output(self):
        """
        generate an output from the token memory machine
        :return:
        """
        pass

    def manage_memory(self):
        """
        manage the memory of the token memory machine
        the memory is managed by removing the least used tokens
        until the memory is within the max_tokens
        :return:
        """
        #we first get the min value from all token usages that we first double

