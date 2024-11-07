import os
import torch
import optuna
from Net import Net

# list all files in models directory
files = os.listdir('models')

study_name = "no-name-f1b1926e-7319-45ff-9b24-d31b65de35e1"
study = optuna.load_study(study_name=study_name, storage="sqlite:///optuna_study.db")

relevant_keys = ["batch_size","memory_size","memory_dim","hidden_dims","use_tanh"]

for fname in files:
    model = torch.load(f'models/{fname}')
    if not "_trial_" in fname:
        print(f"{fname} is not from the optuna study")
        continue
    trial_number = int(fname.split('_')[2].split('.')[0])
    # get the trial from the study
    trial = study.trials[trial_number]
    # get the params from the trial
    params = trial.params
    net_dict = {k: v for k, v in params.items() if k in relevant_keys}
    net_dict["hidden_dims"] = [params["hidden_dims"]]*params["num_layers"]
    net_dict["vocab_size"] = 256
    print(net_dict)
    net = Net(**net_dict)
    net.memory_mlp.load_state_dict(model["memory_mlp"]) #fucked it up with missing those lines - replaced the models with ass shit lmao
    net.pred_mlp.load_state_dict(model["pred_mlp"])
    net.save(f'models/{fname}') # overwrite the model with the new one