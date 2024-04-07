import torch
import sys

from lib.Net import Net

from lib.encoding import encode_string, decode_string, sample_from_logits

net = Net.load(sys.argv[1], batch_size=1, device='cpu')

input_string = sys.argv[2] if len(sys.argv) > 2 else "test "

n_predictions = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

encoded_input = encode_string(input_string, device='cpu', truncate=False)

all_onehots = torch.zeros((encoded_input.shape[0] + n_predictions + 1, 256), device='cpu')

j = 0

for i in range(encoded_input.shape[0]):
    all_onehots[j] = encoded_input[i]
    output = net.forward(encoded_input[i].unsqueeze(0))
    j += 1

for i in range(n_predictions):
    j += 1
    logits = net.forward(all_onehots[j-1].unsqueeze(0))
    all_onehots[j] = sample_from_logits(logits, temperature)

print(decode_string(all_onehots))
