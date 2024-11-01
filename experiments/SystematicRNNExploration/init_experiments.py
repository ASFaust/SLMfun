import json
from itertools import product
import os
import hashlib

from huggingface_hub import file_exists


def rnn_layer_config_generator():
    """
    Generator that yields all valid combinations of initialization arguments
    for the RNNLayer, excluding invalid configurations.

    Yields:
    -------
    config : dict
        Dictionary containing a valid combination of initialization arguments.
    """
    use_input_gate_options = [True, False]
    use_forget_gate_options = [True, False]
    use_output_gate_options = [True, False]
    use_state_gate_options = [True, False]
    use_new_state_options = [True, False]
    use_old_state_options = [True, False]
    use_residual_options = [True, False]

    for (use_input_gate, use_forget_gate, use_output_gate,
         use_state_gate, use_new_state, use_old_state, use_residual) in product(
            use_input_gate_options, use_forget_gate_options,
            use_output_gate_options, use_state_gate_options, use_new_state_options,
            use_old_state_options, use_residual_options):

        # Skip invalid configurations
        if not (use_new_state or use_old_state):
            continue  # At least one of use_new_state or use_old_state must be True

        config = {
            "use_input_gate": use_input_gate,
            "use_forget_gate": use_forget_gate,
            "use_output_gate": use_output_gate,
            "use_state_gate": use_state_gate,
            "use_new_state": use_new_state,
            "use_old_state": use_old_state,
            "use_residual": use_residual
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
    state_sizes = [32, 256]
    for config in rnn_layer_config_generator():
        for n_layers, state_size in product([1, 2, 3], state_sizes):
            new_config = config.copy()
            new_config["n_layers"] = n_layers
            new_config["state_size"] = state_size

            if state_size < 256 and not new_config["use_residual"]:
                #we dont want to test this configuration
                #because having less states than the vocabulary size
                #is not useful without a residual connection
                continue

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

