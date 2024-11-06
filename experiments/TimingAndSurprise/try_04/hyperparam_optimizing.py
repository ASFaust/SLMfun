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
training_time_per_trial = 60 * 5 # 5 minutes
log_frequency = 50


# Define the objective function for Optuna
def objective(trial):
    # Hyperparameter search space
    batch_size =trial.suggest_int("batch_size", 32, 512, log=True)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    memory_size = trial.suggest_int("memory_size", 4, 16, log=True)
    memory_dim = trial.suggest_int("memory_dim", 32, 256, log=True)
    hidden_dims = trial.suggest_int("hidden_dims", 128, 2048, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    label_smoothing = trial.suggest_float("label_smoothing", 1e-8, 1e-1, log=True)
    use_tanh = trial.suggest_categorical("use_tanh", [True, False])

    # Print the hyperparameters
    print(f"Trial {trial.number}")
    print(f"batch_size: {batch_size}")
    print(f"lr: {lr}")
    print(f"memory_size: {memory_size}")
    print(f"memory_dim: {memory_dim}")
    print(f"hidden_dims: {hidden_dims}")
    print(f"num_layers: {num_layers}")
    print(f"label_smoothing: {label_smoothing}")
    print(f"use_tanh: {use_tanh}")

    # Data Generator
    generator = DataGenerator(
        '../../../datasets/enwik8.txt',
        batch_size=batch_size,
        history_size=history_size,
        device=device
    )

    # Model Definition
    net_params = {
        'batch_size': batch_size,
        'vocab_size': 256,
        'memory_size': memory_size,
        'memory_dim': memory_dim,
        'hidden_dims': [hidden_dims] * num_layers,
        'use_tanh': use_tanh
    }
    # save the params in the trial
    trial.set_user_attr("net_params", net_params)

    net = Net(**net_params, device=device)

    # Loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Advanced NaN/Inf loss handling: retry logic
    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        i = 0
        start_time = time.time()
        ma_loss = 3.0  # Initialize moving average loss
        ma_loss_2 = 3.0  # Initialize slower moving average loss for final quality assessment

        # Training Loop
        while (time.time() - start_time) < training_time_per_trial:
            x, _ = generator.get_batch()
            for j in range(x.shape[1] - 1):
                i += 1
                inputs = x[:, j].to(device)
                targets = x[:, j + 1].argmax(dim=1).to(device)
                pred = net.forward(inputs)
                loss = loss_fn(pred, targets)

                if torch.isnan(loss) or torch.isinf(loss) or (loss > 1e5):
                    print(
                        f"\nEncountered NaN/Inf loss at step {i}, retrying trial ({retry_count + 1}/{max_retries})...")
                    retry_count += 1
                    net = Net(**net_params, device=device)  # Reset the model
                    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # Reset the optimizer
                    break  # Exit the inner loop to restart the trial

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_value_(net.parameters(), 1.0)
                optimizer.step()

                ma_loss = 0.95 * ma_loss + 0.05 * loss.item()
                ma_loss_2 = 0.99 * ma_loss_2 + 0.01 * loss.item()

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
        else:
            # If training completes without NaN/Inf, exit retry loop
            break
    else:
        # If maximum retries are reached, stop the trial
        print("\nMaximum retries reached. Stopping trial.")
        return np.inf  # Return a large value to indicate failure

    # Save the model for the best trial
    model_path = f"models/net_trial_{trial.number}.pt"
    torch.save(net.state_dict(), model_path)

    return ma_loss_2


# Run Optuna study
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", storage="sqlite:///optuna_study.db", load_if_exists=True)
    study.optimize(objective, timeout=60 * 60 * 6)  # 6 hours timeout

    # Save study results
    with open("optuna_study_results.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    print("\nBest trial:")
    print(study.best_trial.params)
