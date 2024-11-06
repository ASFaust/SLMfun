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
training_steps = 30000  # Limit each trial to 5000 steps for quicker experimentation
log_frequency = 50

# Define the objective function for Optuna

def objective(trial):
    # Hyperparameter search space
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    memory_size = trial.suggest_categorical("memory_size", [4, 8, 16])
    memory_dim = trial.suggest_categorical("memory_dim", [32, 64, 128, 256])
    hidden_dims = trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024, 2048])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    label_smoothing = trial.suggest_loguniform("label_smoothing", 1e-10, 0.1)

    # Print the hyperparameters
    print(f"Trial {trial.number}")
    print(f"batch_size: {batch_size}")
    print(f"lr: {lr}")
    print(f"memory_size: {memory_size}")
    print(f"memory_dim: {memory_dim}")
    print(f"hidden_dims: {hidden_dims}")
    print(f"num_layers: {num_layers}")
    print(f"label_smoothing: {label_smoothing}")

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
        'hidden_dims': [hidden_dims] * num_layers
    }
    net = Net(**net_params, device=device)

    # Loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Training Loop
    i = 0
    ma_loss = 3.0  # Initialize moving average loss
    while i < training_steps:
        x, _ = generator.get_batch()
        for j in range(x.shape[1] - 1):
            i += 1
            inputs = x[:, j].to(device)
            targets = x[:, j + 1].argmax(dim=1).to(device)
            pred = net.forward(inputs)
            loss = loss_fn(pred, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                print("\nEncountered NaN/Inf loss, stopping trial early.")
                return np.inf  # Return a large value to indicate failure

            optimizer.zero_grad()
            loss.backward()
            ma_loss = 0.95 * ma_loss + 0.05 * loss.item()
            optimizer.step()

            if i % log_frequency == 0:
                print(f"\rStep {i}/{training_steps}, Moving Average Loss: {ma_loss:.3f}", end='', flush=True)

            # Break the inner loop if training steps exceeded
            if i >= training_steps:
                break

        # Reset memory at the end of sequence
        net.reset()

    # Save the model for the best trial
    model_path = f"models/net_trial_{trial.number}.pt"
    torch.save(net.state_dict(), model_path)

    # Return the final moving average loss as the objective value
    return ma_loss

# Run Optuna study
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", storage="sqlite:///optuna_study.db", load_if_exists=True)
    study.optimize(objective, timeout=60*60*6)  # 6 hours timeout

    # Save study results
    with open("optuna_study_results.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    print("\nBest trial:")
    print(study.best_trial.params)
