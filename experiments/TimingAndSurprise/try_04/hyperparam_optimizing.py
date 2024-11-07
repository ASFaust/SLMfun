import optuna
from llvmlite.binding import initialize
from transformers import TransfoXLLMHeadModel

from DataGenerator import DataGenerator
from Net import Net
import torch
import time
import json
import os
import numpy as np
from itertools import product

# Set device and common parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
history_size = 2000
training_time_per_trial = 60 * 5 # 5 minutes of training per trial
log_frequency = 50
fail_loss = 200.0  # Loss value to indicate failure


def get_hyperparameters(trial):
    """
    Define the hyperparameter search space for the Optuna trial.
    """
    return {
        'batch_size': trial.suggest_int("batch_size", 32, 512, log=True),
        'lr': trial.suggest_float("lr", 1e-4, 1e-1, log=True), # Learning rate from 0.0001 to 0.1
        'memory_size': trial.suggest_int("memory_size", 4, 32, log=True),
        'memory_dim': trial.suggest_int("memory_dim", 128, 1024, log=True),
        'hidden_dims': trial.suggest_int("hidden_dims", 512, 4096, log=True),
        'num_layers': trial.suggest_int("num_layers", 1, 3),
        'use_tanh': trial.suggest_categorical("use_tanh", [True, False])
    }


def create_data_generator(batch_size):
    """
    Create a data generator instance.
    """
    return DataGenerator(
        '../../../datasets/enwik8.txt',
        batch_size=batch_size,
        history_size=history_size,
        device=device
    )


def create_model(net_params):
    """
    Create a new instance of the model.
    """
    model_params = {
        'batch_size': net_params['batch_size'],
        'vocab_size': 256,
        'memory_size': net_params['memory_size'],
        'memory_dim': net_params['memory_dim'],
        'hidden_dims': [net_params['hidden_dims']] * net_params['num_layers'],
        'use_tanh': net_params['use_tanh']
    }
    return Net(**model_params, device=device)


def handle_nan_inf_loss(trial, net_params):
    """
    Handle NaN/Inf loss by resetting the model and optimizer.
    """
    print(f"\nEncountered NaN/Inf loss, retrying trial")
    net = create_model(net_params)  # Reset the model
    optimizer = torch.optim.Adam(net.parameters(), lr=net_params["lr"])  # Reset the optimizer
    return net, optimizer


def train_step(net, loss_fn, optimizer, inputs, targets):
    """
    Perform a single training step.
    """
    pred = net.forward(inputs)
    loss = loss_fn(pred, targets)

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping - we dont need this for now
    # torch.nn.utils.clip_grad_value_(net.parameters(), 1.0)
    optimizer.step()

    return loss


def train_epoch(generator, net, loss_fn, optimizer, start_time, ma_loss):
    """
    Train the model for one epoch within the time limit.
    """
    i = 0
    while (time.time() - start_time) < training_time_per_trial:
        x, targets = generator.get_batch()
        for j in range(x.shape[1] - 1):
            i += 1
            loss = train_step(net, loss_fn, optimizer, x[j],  targets[j])

            if torch.isnan(loss) or torch.isinf(loss) or (loss.item() > 1e5) or (ma_loss > fail_loss):
                return i, loss, False  # Indicate failure

            loss_value = loss.item()
            ma_loss = 0.999 * ma_loss + 0.001 * loss_value

            if i % log_frequency == 0:
                time_since_start = time.time() - start_time
                remaining_time = (training_time_per_trial - time_since_start)
                print(
                    f"\rStep {i}, Moving Average Loss 1: {ma_loss:.3f}, Remaining Time: {remaining_time:.1f} seconds",
                    end='', flush=True)

            # Break the inner loop if training time exceeds the limit
            if (time.time() - start_time) > training_time_per_trial:
                break

        # Reset memory at the end of sequence
        net.reset()
    return i, ma_loss, True  # Indicate success


def train_model(trial, generator, net, loss_fn, optimizer):
    """
    Train the model within the allotted time per trial.
    Handle NaN/Inf loss scenarios with retries.
    """
    wasted_time = 0
    n_fails = 0
    ma_loss = 3.0  # Initialize moving average loss

    while True:
        start_time = time.time()
        i, final_loss, success = train_epoch(generator, net, loss_fn, optimizer, start_time, ma_loss)

        if success:
            #save the model
            model_path = f"models/net_trial_{trial.number}.pt"
            torch.save(net.state_dict(), model_path)
            return final_loss
        else:
            n_fails += 1
            if n_fails > 10:
                print(f"\nTrial {trial.number} failed due to retry limit")
                break
            wasted_time += (time.time() - start_time)
            if wasted_time > training_time_per_trial:
                print(f"\nTrial {trial.number} failed due to time limit")
                break
            print(f"\nTrial {trial.number} failed. Total time wasted so far: {wasted_time:.2f} seconds")
            print(f"Time willing to waste yet: {training_time_per_trial - wasted_time:.2f} seconds")
            net, optimizer = handle_nan_inf_loss(trial, trial.user_attrs["net_params"])
            ma_loss = 3.0

    print(f"\nTrial {trial.number} failed")
    return fail_loss


def objective(trial):
    """
    Objective function for Optuna to optimize.
    """
    # Get hyperparameters
    hyperparams = get_hyperparameters(trial)
    trial.set_user_attr("net_params", hyperparams)

    # Print the hyperparameters
    print(f"Trial {trial.number}")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")

    # Data Generator
    generator = create_data_generator(hyperparams['batch_size'])

    # Model Definition
    net = create_model(hyperparams)

    # Loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'])

    # Train the model
    final_loss = train_model(trial, generator, net, loss_fn, optimizer)

    return final_loss


def create_initial_trials(study):
    initial_params = []
    useful_batch_sizes = [256]
    useful_memory_sizes = [8]
    useful_memory_dims = [256]
    useful_hidden_dims = [1024]
    useful_num_layers = [2]
    useful_use_tanh = [False]
    useful_lr = [0.001]
    for batch_size, memory_size, memory_dim, hidden_dims, num_layers, use_tanh, lr in product(
            useful_batch_sizes, useful_memory_sizes, useful_memory_dims, useful_hidden_dims, useful_num_layers, useful_use_tanh, useful_lr):
        if hidden_dims < memory_dim:
            continue # skip this, because memory_dim < hidden_dims is not logical
        initial_params.append({
            'batch_size': batch_size,
            'lr': lr,
            'memory_size': memory_size,
            'memory_dim': memory_dim,
            'hidden_dims': hidden_dims,
            'num_layers': num_layers,
            'use_tanh': use_tanh
        })

    # Get completed trial parameters to avoid duplicates
    completed_params = [{k: v for k, v in trial.params.items()} for trial in study.trials]

    # Filter initial parameters to only those not in completed trials
    params_to_enqueue = [p for p in initial_params if p not in completed_params]

    print(f"Enqueuing {len(params_to_enqueue)} initial trials...")

    # Enqueue each parameter set individually
    for params in params_to_enqueue:
        study.enqueue_trial(params)

    print("Initial trials enqueued.")
    return study


def main():
    """
    Main function to run the Optuna study.
    """
    study = optuna.create_study(direction="minimize", storage="sqlite:///optuna_study.db", load_if_exists=True, study_name="no-name-f1b1926e-7319-45ff-9b24-d31b65de35e1")
    study = create_initial_trials(study)
    study.optimize(objective, timeout=60 * 60 * 12)  # 12 hours timeout

    # Save study results
    with open("optuna_study_results.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    print("\nBest trial:")
    print(study.best_trial.params)


if __name__ == "__main__":
    main()
