import optuna
from DataGenerator import DataGenerator
from Net import Net
import torch
import time
import json
import os
import numpy as np

# Set device and common parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
history_size = 2000
training_time_per_trial = 20 # 20 seconds - just to test the code
log_frequency = 50


def get_hyperparameters(trial):
    """
    Define the hyperparameter search space for the Optuna trial.
    """
    return {
        'batch_size': trial.suggest_int("batch_size", 32, 512, log=True),
        'lr': trial.suggest_float("lr", 1e-4, 1e-1, log=True), # Learning rate from 0.0001 to 0.1
        'memory_size': trial.suggest_int("memory_size", 4, 16, log=True),
        'memory_dim': trial.suggest_int("memory_dim", 32, 256, log=True),
        'hidden_dims': trial.suggest_int("hidden_dims", 128, 2048, log=True),
        'num_layers': trial.suggest_int("num_layers", 1, 3),
        'label_smoothing': trial.suggest_float("label_smoothing", 1e-8, 1e-1, log=True),
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


def handle_nan_inf_loss(trial, retry_count, max_retries, net_params):
    """
    Handle NaN/Inf loss by resetting the model and optimizer.
    """
    print(f"\nEncountered NaN/Inf loss, retrying trial ({retry_count + 1}/{max_retries})...")
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


def train_epoch(generator, net, loss_fn, optimizer, start_time, ma_loss, ma_loss_2):
    """
    Train the model for one epoch within the time limit.
    """
    i = 0
    while (time.time() - start_time) < training_time_per_trial:
        x, _ = generator.get_batch()
        for j in range(x.shape[1] - 1):
            i += 1
            inputs = x[:, j].to(device)
            targets = x[:, j + 1].argmax(dim=1).to(device)
            loss = train_step(net, loss_fn, optimizer, inputs, targets)

            if torch.isnan(loss) or torch.isinf(loss) or (loss.item() > 1e5) or (ma_loss_2 > 200):
                return i, loss, False  # Indicate failure

            loss_value = loss.item()
            ma_loss = 0.95 * ma_loss + 0.05 * loss_value
            ma_loss_2 = 0.99 * ma_loss_2 + 0.01 * loss_value

            if i % log_frequency == 0:
                time_since_start = time.time() - start_time
                remaining_time = (training_time_per_trial - time_since_start)
                print(
                    f"\rStep {i}, Moving Average Loss 1: {ma_loss:.3f}, Moving Average Loss 2: {ma_loss_2:.3f}, Remaining Time: {remaining_time:.1f} seconds",
                    end='', flush=True)

            # Break the inner loop if training time exceeds the limit
            if (time.time() - start_time) > training_time_per_trial:
                break

        # Reset memory at the end of sequence
        net.reset()
    return i, ma_loss_2, True  # Indicate success


def train_model(trial, generator, net, loss_fn, optimizer):
    """
    Train the model within the allotted time per trial.
    Handle NaN/Inf loss scenarios with retries.
    """
    max_retries = 5
    retry_count = 0
    ma_loss = 3.0  # Initialize moving average loss
    ma_loss_2 = 3.0  # Initialize slower moving average loss for final quality assessment

    while retry_count < max_retries:
        start_time = time.time()
        i, final_loss, success = train_epoch(generator, net, loss_fn, optimizer, start_time, ma_loss, ma_loss_2)

        if success:
            return final_loss
        else:
            retry_count += 1
            net, optimizer = handle_nan_inf_loss(trial, retry_count, max_retries, trial.user_attrs["net_params"])
            ma_loss = 3.0
            ma_loss_2 = 3.0

    # If maximum retries are reached, stop the trial
    print("\nMaximum retries reached. Stopping trial.")
    return np.inf  # Return a large value to indicate failure


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
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=hyperparams['label_smoothing'])
    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'])

    # Train the model
    final_loss = train_model(trial, generator, net, loss_fn, optimizer)

    # Save the model for the best trial
    model_path = f"models/net_trial_{trial.number}.pt"
    torch.save(net.state_dict(), model_path)

    return final_loss


def main():
    """
    Main function to run the Optuna study.
    """
    study = optuna.create_study(direction="minimize", storage="sqlite:///optuna_study.db", load_if_exists=True)
    study.optimize(objective, timeout=60 * 60 * 6)  # 6 hours timeout

    # Save study results
    with open("optuna_study_results.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    print("\nBest trial:")
    print(study.best_trial.params)


if __name__ == "__main__":
    main()
