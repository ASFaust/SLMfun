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

input_string = sys.argv[2]

n_predictions = int(sys.argv[3])

temperature = float(sys.argv[4])

encoded_input = encode_string(input_string, device, truncate=False)

all_onehots = torch.zeros((encoded_input.shape[0] + n_predictions + 1, 256), device=device)

j = 0

for i in range(encoded_input.shape[0]):
    all_onehots[j] = encoded_input[i]
    output = net.forward(encoded_input[i].unsqueeze(0))
    j += 1

all_onehots[j] = sample_from_logits(output, temperature)

for i in range(n_predictions):
    j += 1
    logits = net.forward(all_onehots[j-1].unsqueeze(0))
    all_onehots[j] = sample_from_logits(logits, temperature)

print(decode_string(all_onehots))