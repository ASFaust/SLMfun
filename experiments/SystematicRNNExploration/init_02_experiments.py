import os
import json
import hashlib

"""
This script will create a new set of configurations based on the best 10% of the results from the first training run
It will then save these configurations to a new file
It will also create a new results folder
The new configs it creates are:
- The best 10% of the results
- The best 10% of the results with a state size of 512
- The best 10% of the results with 3 layers and a state size of 256
- The best 10% of the results with 3 layers and a state size of 512
"""

results_folder = "results_01_all"

if not os.path.exists(results_folder):
    raise Exception("The results folder does not exist. Please run the training script first. see Readme.md")

results = os.listdir(results_folder)

results = [r for r in results if r.endswith(".json")]

results = [json.load(open(f"{results_folder}/{r}")) for r in results]

print(f"Number of results: {len(results)}")

# Sort the results by the key "min loss"

results = sorted(results, key=lambda x: x["min loss"])

#get the highest 10% of the results

top_10_percent = results[:int(len(results) * 0.1)]

new_configs = []

to_copy = ["use_input_gate", "use_forget_gate", "use_output_gate", "use_state_gate", "use_new_state", "use_old_state", "use_residual", "n_layers", "state_size"]

def get_id(r):
    config_str = json.dumps(r, sort_keys=True).encode('utf-8')
    id = hashlib.sha256(config_str).hexdigest()
    return id

def create_new_config(r, state_size=None, n_layers=None):
    new_config = {}
    new_config["old id"] = r["id"]
    for key in to_copy:
        new_config[key] = r[key]
    if state_size is not None:
        new_config["state_size"] = state_size
    if n_layers is not None:
        new_config["n_layers"] = n_layers
    new_config["id"] = get_id(new_config)
    return new_config

for r in top_10_percent:
    new_config = {}
    new_config["old id"] = r["id"]
    for key in to_copy:
        new_config[key] = r[key]
    new_config["id"] = get_id(new_config)
    new_configs.append(new_config) #original config
    new_configs.append(create_new_config(r, state_size=512)) #512 states
    if new_config["n_layers"] < 3:
        new_configs.append(create_new_config(r, n_layers=3, state_size=256)) #3 layers, 256 states
        new_configs.append(create_new_config(r, n_layers=3, state_size=512)) #3 layers, 512 states

print(f"Number of new configs: {len(new_configs)}")

with open("rnn_configs_02.json", "w") as f:
    json.dump(new_configs, f, indent=4)

#also create a new results folder
results_folder = "results_02_big"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)