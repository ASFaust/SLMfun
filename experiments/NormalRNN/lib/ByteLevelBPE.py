import torch
import json

class ByteLevelBPE:
    def __init__(self,decode_dict,encoding,vocab_size):
        self.decode_dict = decode_dict
        self.encoding = encoding
        self.vocab_size = vocab_size

    def encode(self,text):
        #returns one-hot encoding of text
        encoded = list(text.encode('utf-8'))

        for enc in self.encoding:
            #enc is the pair of integers that we are replacing
            new_encoded = []
            j = 0
            while j < (len(encoded)-1):
                if encoded[j] == enc[0] and encoded[j+1] == enc[1]:
                    new_encoded.append(i)
                    j += 2
                else:
                    new_encoded.append(encoded[j])
                    j += 1
            encoded = new_encoded

        one_hot = torch.zeros(len(encoded),self.vocab_size)
        for i in range(len(encoded)):
            one_hot[i,encoded[i]] = 1.0
        return one_hot

    def sample(self,logits,temperature=1.0,top_k=None):
        #logits is a tensor of shape (1,vocab_size)
        #returns a one-hot encoding of the sampled token
        if temperature == 0.0:
            return torch.argmax(logits)
        else:
            probs = torch.softmax(logits/temperature,dim=1)
            if top_k is not None:
                probs = torch.topk(probs,top_k)[0]
                probs = probs / torch.sum(probs)
            return torch.multinomial(probs,1)

    def decode(self,encoded_tensor):
        #encoded_tensor is a tensor of shape (seq_len,vocab_size)
        #where each row is a one-hot encoding of a token
        #returns a string
        encoded = torch.argmax(encoded_tensor,dim=1)
        #we can use the decode_dict to convert the encoded tensor back to bytes.
        #then we can use the bytes.decode() method to convert back to a string.
        bytes = []
        for i in range(len(encoded)):
            if encoded[i] in self.decode_dict:
                #decode_dict[encoded[i]] is a list of bytes
                bytes.extend(self.decode_dict[encoded[i]])
            else:
                bytes.append(encoded[i])
        #decode bytes to string unicode and ignore errors
        return bytes.decode('utf-8',errors='ignore')

    def save(self,path):
        with open(path,'w') as f:
            json.dump(
                    {
                        'decode_dict':self.decode_dict,
                        'encoding':self.encoding,
                        'vocab_size':self.vocab_size
                    },f,indent=4)

    @staticmethod
    def load(path):
        with open(path,'r') as f:
            data = json.load(f)
            return ByteLevelBPE(data['decode_dict'],data['encoding'],data['vocab_size'])

    @staticmethod
    def learn(text,vocab_size=1024):

        #these are the values passed to the constructor of BPE:
        decode_dict = {} #maps from additional tokens to the bytes they represent
        encoding = [] #holds the things that need to be replaced in each step

        bytes = list(text.encode('utf-8'))
        pair_counts = torch.zeros(256,256)
        encoded = []
        for i in range(len(bytes)-1):
            print(f"\r{i/len(bytes)*100.0:.3f}%",end='',flush=True)
            pair_counts[bytes[i],bytes[i+1]] += 1
            encoded.append(int(bytes[i]))
            #print(big_string)
        #now we have a matrix of counts, we need to find the most common pairs
        # Find the maximum value and its indices
        n_tokens = 256
        while n_tokens < vocab_size:
            max_value = pair_counts.max()  # Find the maximum value
            print("max value:",max_value)
            max_indices = torch.nonzero(pair_counts == max_value)[0]  # Find the indices of the maximum value
            to_replace = [int(max_indices[0]),int(max_indices[1])]

            encoding.append(to_replace)
            decode_dict[n_tokens] = to_replace
            if decode_dict[n_tokens][0] > 255:
                decode_dict[n_tokens] = decode_dict[decode_dict[n_tokens][0]] + [decode_dict[n_tokens][1]]
            if decode_dict[n_tokens][-1] > 255:
                decode_dict[n_tokens] = decode_dict[n_tokens][:-1] + decode_dict[decode_dict[n_tokens][-1]]

            new_pair_counts = torch.zeros(n_tokens+1,n_tokens+1)
            new_pair_counts[:n_tokens,:n_tokens] = pair_counts
            new_pair_counts[max_indices[0],max_indices[1]] = 0
            #there is one more thing to do: count the total occurrences of both original tokens
            #after they have been replaced by the new token
            #it could be that either one or both of them completely disappear from the text
            #so that we could delete them from the token list, reduce the vocab size and make space for a new token
            new_encoded = []
            i = 0
            while i < (len(encoded) - 1):
                if encoded[i] == to_replace[0] and encoded[i+1] == to_replace[1]:
                    new_encoded.append(n_tokens)
                    #also update the counts
                    if i > 0:
                        new_pair_counts[encoded[i-1],n_tokens] += 1
                    if i < (len(encoded) - 2):
                        new_pair_counts[n_tokens,encoded[i+2]] += 1
                    i += 1
                else:
                    new_encoded.append(encoded[i])
                i += 1
            print("length contrast:",len(encoded),len(new_encoded))
            encoded = new_encoded
            pair_counts = new_pair_counts
            n_tokens += 1
        return ByteLevelBPE(decode_dict,encoding,vocab_size)

