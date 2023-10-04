from encoding import encode_string, decode_string
from lstm import Net
import torch
import time
import sys

def sample_from_logits(logits, temperature):
    if temperature == 0:
        return make_onehot_from_index(torch.argmax(logits))
    logits = logits.reshape(1, 256)
    logits = logits / temperature
    probabilities = torch.softmax(logits, dim=1)
    ret = torch.multinomial(probabilities, 1)[0,0]
    ret = make_onehot_from_index(ret)
    return ret

#python3 speak.py models/model_nietz.pt "wernicht" 1000

def make_onehot_from_index(index):
    onehot = torch.zeros(256)
    onehot[index] = 1
    return onehot

history_size = 1
device = 'cpu'

net = Net.load(sys.argv[1], batch_size=1, device=device)

input_string = sys.argv[2] + "\n"

temperature = float(sys.argv[3])

encoded_input = encode_string(input_string, device, truncate=False)

all_onehots = [] #torch.zeros((encoded_input.shape[0] + 1, 256), device=device)

for i in range(encoded_input.shape[0]):
    all_onehots.append(encoded_input[i])
    output = net.forward(encoded_input[i].unsqueeze(0))

all_onehots.append(sample_from_logits(output, temperature))

for i in range(1000):
    logits = net.forward(all_onehots[-1].unsqueeze(0))
    all_onehots.append(sample_from_logits(logits, temperature))
    if decode_string(all_onehots[-1].unsqueeze(0)) == "\n":
        break

#make all_onehots into a tensor
all_onehots = torch.stack(all_onehots)

print(decode_string(all_onehots))

while True:
    next_line = input()
    if next_line == "":
        break
    encoded_input = encode_string(next_line + "\n", device, truncate=False)
    all_onehots = []
    for i in range(encoded_input.shape[0]):
        all_onehots.append(encoded_input[i])
        output = net.forward(encoded_input[i].unsqueeze(0))
    all_onehots.append(sample_from_logits(output, temperature))
    for i in range(1000):
        logits = net.forward(all_onehots[-1].unsqueeze(0))
        all_onehots.append(sample_from_logits(logits, temperature))
        if decode_string(all_onehots[-1].unsqueeze(0)) == "\n":
            break
    #make all_onehots into a tensor
    all_onehots = torch.stack(all_onehots)
    print(decode_string(all_onehots))
