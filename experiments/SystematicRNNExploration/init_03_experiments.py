import os
import json
import hashlib

"""
This script will create a new set of configurations based on the best 10 of the results from the second training run
It will then save these configurations to a new file.
Each configuration is run 10 times. 
"""

results_folder = "results_02_big"

if not os.path.exists(results_folder):
    raise Exception("The results folder does not exist. Please run the training script first. see Readme.md")

results = os.listdir(results_folder)

results = [r for r in results if r.endswith(".json")]

results = [json.load(open(f"{results_folder}/{r}")) for r in results]

print(f"Number of results: {len(results)}")

# Sort the results by the key "min loss"

results = sorted(results, key=lambda x: x["min loss"])

#get the highest 10% of the results

top_10 = results[:10]

to_copy = ["use_input_gate", "use_forget_gate", "use_output_gate", "use_state_gate", "use_new_state", "use_old_state", "use_residual", "n_layers", "state_size"]

def get_id(r):
    config_str = json.dumps(r, sort_keys=True).encode('utf-8')
    id = hashlib.sha256(config_str).hexdigest()
    return id

def create_new_config(config, i):
    new_config = {}
    new_config["run 1 id"] = config["old id"]
    new_config["run 2 id"] = config["id"]
    for key in to_copy:
        new_config[key] = config[key]
    new_config["common id"] = get_id(new_config)
    new_config["training"] = i
    new_config["unique id"] = get_id(new_config)
    return new_config

new_configs = []

for r in top_10:
    for i in range(10):
        new_configs.append(create_new_config(r, i))

print(f"Number of new configs: {len(new_configs)}")

with open("rnn_configs_03.json", "w") as f:
    json.dump(new_configs, f, indent=4)

#also create a new results folder
results_folder = "results_03"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
