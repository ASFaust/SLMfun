import matplotlib.pyplot as plt
import json
import os
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# Set Seaborn style for aesthetics
sns.set_theme(style="whitegrid")

# Load the results
results = []

folder = "results_02_big"
files = os.listdir(folder)

for file in files:
    with open(os.path.join(folder, file), 'r') as f:
        results.append(json.load(f))

# Plot the results
plt.figure(figsize=(14, 7))

best_min_loss = float('inf')
best_min_loss_result = None

cleaned_results = [r for r in results if 'loss' in r]

# Apply Gaussian smoothing and plot each result
for result in cleaned_results:
    smoothed_loss = gaussian_filter1d(result['loss'], sigma=6)  # Smoothing parameter
    plt.plot(smoothed_loss, label=f"Min loss: {result['min loss']:.4f}")
    if result['min loss'] < best_min_loss:
        print(f"New best loss: {result['min loss']}")
        best_min_loss = result['min loss']
        best_min_loss_result = result

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('Smoothed Training Loss Curves (Log Scale)')
plt.legend(loc='upper right')
plt.tight_layout()

# Highlight the best result
print("Best result:")
print_result = best_min_loss_result.copy()
del print_result['loss']
print(json.dumps(print_result, indent=4))

# Save the high-resolution plot
plt.savefig('results_02_big_highres.png', dpi=300)

hyperparameters = ['state_size', 'use_input_gate', 'use_forget_gate', 'use_output_gate', 'use_state_gate', 'use_new_state', 'use_old_state', 'use_residual', 'n_layers']

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Loop through each hyperparameter and plot with seaborn swarmplot
for i, hyperparameter in enumerate(hyperparameters):
    values = [r[hyperparameter] for r in cleaned_results]
    losses = [r['min loss'] for r in cleaned_results]

    # filter out losses > 1.9:
    values = [v for v, loss in zip(values, losses) if loss < 1.9]
    losses = [loss for loss in losses if loss < 1.9]

    # Create the swarm plot
    plt.subplot(3, 3, i + 1)

    # Get unique values and calculate mean loss for each unique value of the hyperparameter
    unique_values = sorted(set(values))
    mean_losses = [np.mean([loss for v, loss in zip(values, losses) if v == val]) for val in unique_values]

    # Use indices for x positions to match categorical plotting
    x_positions = range(len(unique_values))  # Categorical index positions
    plt.scatter(x_positions, mean_losses, color='red', s=50, label='Mean', zorder=1)

    sns.swarmplot(x=values, y=losses, size=4, alpha=0.7)


    # Ensure x-ticks and labels match categorical values
    plt.xticks(ticks=x_positions, labels=unique_values)

    plt.xlabel(hyperparameter)
    plt.ylabel('Final Loss')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig('results_02_big_hyperparameters.png')

#make an estimate how long the training will take
#we do that by taking the keys "start time" and "end time" and calculate the difference
#and we calculate the average of that
#and then we load the rnn_configs_02.json file and subtract len(cleaned_results) from the number of configurations
#and multiply that with the average time

# Load the configurations
with open('rnn_configs_02.json', 'r') as f:
    rnn_configs = json.load(f)

# Calculate the average training time
total_time = 0
for result in cleaned_results:
    total_time += result['end time'] - result['start time']

average_time = total_time / len(cleaned_results)

# Calculate the remaining training time
remaining_configs = len(rnn_configs) - len(cleaned_results)

n_processes = 2 # Number of parallel processes for training

print(f"Number of remaining configurations: {remaining_configs}")
remaining_time = remaining_configs * average_time / n_processes

average_time_minutes = int(average_time // 60)
average_time_seconds = int(average_time % 60)

print(f"Average training time per configuration: {average_time_minutes} minutes, {average_time_seconds} seconds")

# Convert remaining time to hours, minutes, and seconds

hours = int(remaining_time // 3600)
minutes = int((remaining_time % 3600) // 60)
seconds = int(remaining_time % 60)

print(f"Estimated remaining training time: {hours} hours, {minutes} minutes, {seconds} seconds, assuming {n_processes} parallel processes")
