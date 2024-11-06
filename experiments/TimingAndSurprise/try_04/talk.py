import sys
from Net import Net
from encoding import encode_string, decode_string, sample_from_logits
import torch

# first argument is the model
# second argument is the text we prime the model with
# third argument is the number of characters to generate
# fourth argument is the temperature
#example: python3 talk.py models/net.pt "article" 1000 0.1

model_path = sys.argv[1]
prime_text = sys.argv[2]
num_chars = int(sys.argv[3])
temp = float(sys.argv[4])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = Net.load(model_path, batch_size=1, device=device)

# prime the model with the given text

prime_tensor = encode_string(prime_text, device)

print(prime_tensor.shape)

for i in range(prime_tensor.shape[0]):

    last_pred = net.forward(prime_tensor[i].unsqueeze(0))

last_pred = sample_from_logits(last_pred, temperature=temp)
# generate the text

generated_tensors = [last_pred]

for i in range(num_chars):
    pred = net.forward(generated_tensors[-1])
    pred = sample_from_logits(pred, temperature=temp)
    generated_tensors.append(pred)

generated_text = prime_text + decode_string(torch.stack(generated_tensors))

print(generated_text)

