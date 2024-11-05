from DataGenerator import DataGenerator
from Net import Net
import torch
import time
import json
import os
import numpy as np


history_size = 2000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256
training_steps = 30000
log_frequency = 100 #how often to save the loss to the json file

generator = DataGenerator(
    '../../datasets/enwik8.txt',
    batch_size=batch_size,
    history_size=history_size,
    device=device
)

with open('init_configs.json', 'r') as f:
    configs = json.load(f)

while True:
    result_ids = []
    for fname in os.listdir('results'):
        with open(f'results/{fname}', 'r') as f:
            result = json.load(f)
            #what do we want to do with the results?
            #we want to get the ids so we can do rejection sampling
            result_ids.append(result['id'])
    if len(result_ids) == len(configs):
        print("All configs have been tried")
        break
    next_config = None
    while next_config is None:
        config = np.random.choice(configs)
        if config['id'] not in result_ids:
            next_config = config

    with open(f'results/{next_config["id"]}.json', 'w') as f:
        json.dump(next_config, f, indent=4)

    net_params = {
        'batch_size': batch_size,
        'vocab_size': 256,
        'memory_size': next_config['memory_size'],
        'hidden_dims': [1024, 1024],
    }
    print("Training with config:", next_config)
    net = Net(**net_params, device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    i = 0
    ma_loss = 3.0
    training_curve = []
    smoothed_training_curve = []
    while i < training_steps:
        x, _ = generator.get_batch()
        for j in range(x.shape[1] - 1):
            i += 1
            inputs = x[:,j].to(device)
            targets = x[:,j+1].argmax(dim=1).to(device)
            pred = net.forward(inputs)
            loss = loss_fn(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            ma_loss = 0.95 * ma_loss + 0.05 * loss.item()
            optimizer.step()
            if i % log_frequency == 0:
                print(f"\r{i}: {ma_loss:.3f}", end='', flush=True)
                training_curve.append(loss.item())
                smoothed_training_curve.append(ma_loss)
        net.reset()
    with open(f'results/{next_config["id"]}.json', 'w') as f:
        next_config['loss'] = ma_loss
        next_config['training_curve'] = training_curve
        next_config['smoothed_training_curve'] = smoothed_training_curve
        next_config["net_params"] = net_params
        json.dump(next_config, f, indent=4)

    #also save the model
    torch.save(net.state_dict(), f'models/{next_config["id"]}.pt')
