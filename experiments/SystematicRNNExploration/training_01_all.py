from DataGenerator import DataGenerator
from RNN import RNN
import torch
import time
#import wandb
import json
import os

"""
in this first training, we train all of the configurations for 10k steps. 
this is to get a first understanding of which configurations are promising
we later will refine the search space based on the results of this first training, which is why 10k steps is enough
"""

history_size = 1000
device = 'cuda'
batch_size = 256
training_steps = 10000
log_frequency = 50 #how often to save the loss to the json file

generator = DataGenerator(
    '../../datasets/enwik8.txt',
    batch_size=batch_size,
    history_size=history_size,
    device=device
)

configs = json.load(open('rnn_configs.json', 'r'))

while True:
    print("starting a new run")

    results_folder_contents = os.listdir("results_01_all")

    if len(results_folder_contents) >= len(configs):
        print("All configurations have been run")
        break

    while True:
        current_config = configs[torch.randint(0, len(configs), (1,)).item()]
        if not (f"{current_config['id']}.json" in results_folder_contents):
            break

    #we chose a config that has not been run yet

    #we now save the config to the results folder
    with open(f"results_01_all/{current_config['id']}.json", "w") as f:
        info_dict = {"start time" : time.time()}
        info_dict.update(current_config)
        json.dump(info_dict, f, indent=4)

    print("running config:")
    print(json.dumps(current_config, indent=4))

    rnn_params = current_config.copy()
    rnn_params["batch_size"] = batch_size
    #delete id from the config
    del rnn_params["id"]

    net = RNN(**rnn_params).to(device)

    #wandb.init(project="Systematic Exploration of RNN architectures")
    #wandb.watch(rnn)

    #wandb.config.update(rnn_params)

    #cross entropy loss that expects a target index and logit output
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    i = 0
    ma_loss = 3.0
    loss_curve = []
    while i < training_steps:
        x, _ = generator.get_batch()
        # x is a tensor of shape (batch_size, history_size, 256)
        for j in range(x.shape[1] - 1):
            i += 1
            inputs = x[:,j].to(device)
            targets = x[:,j+1].argmax(dim=1).to(device)
            pred = net.forward(inputs)
            loss = loss_fn(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            ma_loss = 0.95 * ma_loss + 0.05 * loss.item()
            print(f"\r{i}: {ma_loss:.3f}", end='', flush=True)
            optimizer.step()

            if i % log_frequency == 0:
                loss_curve.append(ma_loss)
        net.reset()

    #save the loss curve to the results folder
    with open(f"results_01_all/{current_config['id']}.json", "r") as f:
        info_dict = json.load(f)
    info_dict["loss"] = loss_curve
    info_dict["min loss"] = min(loss_curve)
    info_dict["training steps"] = training_steps
    info_dict["batch size"] = batch_size
    info_dict["history size"] = history_size
    info_dict["dataset"] = "enwik8"
    info_dict["end time"] = time.time()
    with open(f"results_01_all/{current_config['id']}.json", "w") as f:
        json.dump(info_dict, f, indent=4)
