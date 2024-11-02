import matplotlib.pyplot as plt
import json
import os
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# Set Seaborn style for aesthetics
sns.set_theme(style="whitegrid")

# Create the output directory if it doesn't exist
output_folder = "examination_results_03"
os.makedirs(output_folder, exist_ok=True)

# Load the results
results = []

folder = "results_03"
files = os.listdir(folder)

for file in files:
    with open(os.path.join(folder, file), 'r') as f:
        results.append(json.load(f))

cleaned_results = [r for r in results if 'loss' in r]

grouped = {}

for result in cleaned_results:
    key = result["common id"]
    if not key in grouped:
        grouped[key] = []
    grouped[key].append(result)

# Extract min losses for each group
min_losses = []

# Group and extract min losses
for key, group in grouped.items():
    min_loss_values = [entry['min loss'] for entry in group if 'min loss' in entry]
    if min_loss_values:
        min_losses.extend(min_loss_values)

# Sort groups by average min loss
avg_min_losses = [(key, np.mean([entry['min loss'] for entry in group if 'min loss' in entry])) for key, group in grouped.items() if 'min loss' in group[0]]
sorted_indices = np.argsort([avg for _, avg in avg_min_losses])
sorted_keys = [avg_min_losses[i][0] for i in sorted_indices]
sorted_numeric_labels = [i + 1 for i in range(len(sorted_keys))]  # Numeric labels from 1 to len(grouped.keys())

# Assign new labels based on sorted order
label_mapping = {original_key: sorted_numeric_labels[i] for i, original_key in enumerate(sorted_keys)}

group_labels = [label_mapping[key] for key in sorted_keys]

# Update min_losses group labels
min_loss_labels = []
for key, group in grouped.items():
    min_loss_values = [entry['min loss'] for entry in group if 'min loss' in entry]
    if min_loss_values:
        min_loss_labels.extend([label_mapping[key]] * len(min_loss_values))

# Create the swarm plot
plt.figure(figsize=(14, 7))
sns.swarmplot(x=min_loss_labels, y=min_losses, order=sorted_numeric_labels, dodge=True)

# Customize the plot
plt.title("Swarm Plot of Min Losses Grouped by Hyperparameter Config")
plt.xlabel("Hyperparameter Config (Numeric ID)")
plt.ylabel("Min Loss")
plt.xticks(ticks=np.arange(0, len(sorted_numeric_labels)), labels=sorted_numeric_labels, ha='center')  # Explicitly specify x-tick labels, adjust alignment
plt.tight_layout()

# Save the high-resolution plot
plt.savefig(os.path.join(output_folder, 'results_03_highres.png'), dpi=300)

# Plot the training curves for each group
for key in sorted_keys:
    plt.figure(figsize=(14, 7))
    group = grouped[key]
    all_losses = np.array([entry['loss'] for entry in group])
    avg_loss = np.mean(all_losses, axis=0)

    # Plot individual runs with thin, faint lines
    for loss in all_losses:
        plt.plot(loss, color='gray', linewidth=0.5, alpha=0.5)

    # Plot average loss with a bold line
    plt.plot(avg_loss, color='blue', linewidth=2.5, label='Average Loss')

    # Customize the plot
    plt.title(f"Training Curves for Group {label_mapping[key]}")
    plt.xlabel("Time Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_folder, f'training_curve_group_{label_mapping[key]}.png'), dpi=300)

# Plot the average training curves for all groups in one plot
plt.figure(figsize=(14, 7))
for key in sorted_keys:
    group = grouped[key]
    all_losses = np.array([entry['loss'] for entry in group])
    avg_loss = np.mean(all_losses, axis=0)
    plt.plot(avg_loss, linewidth=2, label=f'Group {label_mapping[key]}')

# Customize the final plot
plt.title("Average Training Curves for All Groups")
plt.xlabel("Time Step")
plt.ylabel("Loss")
plt.legend(title="Hyperparameter Configs", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()

# Save the final plot
plt.savefig(os.path.join(output_folder, 'average_training_curves_all_groups.png'), dpi=300)

# Create a markdown file to save the configs of each group
with open(os.path.join(output_folder, 'group_configs.md'), 'w') as md_file:
    for key in sorted_keys:
        group_label = label_mapping[key]
        config = grouped[key][0]  # Assuming all entries in the group have the same config
        filtered_config = {k: config[k] for k in [
            "use_input_gate", "use_forget_gate", "use_output_gate", "use_state_gate",
            "use_new_state", "use_old_state", "use_residual", "n_layers", "state_size"
        ] if k in config}
        md_file.write(f"# Group {group_label}:\n")
        avg_min_loss = np.mean([entry['min loss'] for entry in grouped[key] if 'min loss' in entry])
        md_file.write(f"Average Min Loss: {avg_min_loss}\n")
        md_file.write("Config:\n")
        md_file.write("```json\n")
        md_file.write(json.dumps(filtered_config, indent=4))
        md_file.write("\n```\n")
