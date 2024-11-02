import json
from itertools import product
import os
import hashlib

def rnn_layer_config_generator():
    early_forget_gate_options = [True, False]
    one_gate_for_all_options = [True, False]
    save_updated_state0_options = [True, False]
    forget_gate_for_state1_options = [True, False]

    for early_forget_gate, one_gate_for_all, save_updated_state0, forget_gate_for_state1 in product(
            early_forget_gate_options,
            one_gate_for_all_options,
            save_updated_state0_options,
            forget_gate_for_state1_options):
        config = {
            "early_forget_gate": early_forget_gate,
            "one_gate_for_all": one_gate_for_all,
            "save_updated_state0": save_updated_state0,
            "forget_gate_for_state1": forget_gate_for_state1
        }

        yield config

if __name__ == "__main__":

    # Define the valid state sizes

    config_file = "rnn_configs.json"
    #test if the file already exists
    file_exists = False
    if os.path.exists(config_file):
        print("file already exists. extending it with new configurations")
        file_exists = True
        with open(config_file, "r") as f:
            all_configs = json.load(f)
    else:
        all_configs = []
    state_sizes = [512]
    for config in rnn_layer_config_generator():
        for n_layers, state_size in product([1, 2, 3], state_sizes):
            new_config = config.copy()
            new_config["n_layers"] = n_layers
            new_config["state_size"] = state_size

            config_str = json.dumps(new_config, sort_keys=True).encode('utf-8')
            id = hashlib.sha256(config_str).hexdigest()
            new_config["id"] = id

            if not new_config in all_configs:
                all_configs.append(new_config)
    print(f"New number of configs: {len(all_configs)}")

    with open(config_file, "w") as f:
        json.dump(all_configs, f, indent=4)

    results_folder = "results_01_all"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    print("Done")

