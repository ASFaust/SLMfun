import json
import os
import numpy as np
import hashlib

#the only values these configs have are the memory sizes
#the rest of the values are hardcoded in the train.py file

max_memory_size = 10
n_tries_per_memory_size = 5
memory_sizes = list(range(1, max_memory_size + 1))

def get_id(config):
    """
    Returns a unique id for a config, as a string
    """
    return hashlib.md5(json.dumps(config).encode()).hexdigest()

configs = []
for memory_size in memory_sizes:
    for i in range(n_tries_per_memory_size):
        config = {
            'memory_size': memory_size,
            'trial': i
        }
        config['id'] = get_id(config)
        configs.append(config)

#save the configs to a file called init_configs.json
with open('init_configs.json', 'w') as f:
    json.dump(configs, f, indent=4)

#create a directory for the resulting informations
if not os.path.exists('results'):
    os.makedirs('results')
