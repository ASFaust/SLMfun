import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import os
import pandas as pd

# First, read all files in the results directory
results = []
for fname in os.listdir('results'):
    with open(f'results/{fname}', 'r') as f:
        result = json.load(f)
        if not 'training_curve' in result:
            continue
        results.append(result)

# Collect memory size and final smoothed loss values for each trial
data = {'memory_size': [], 'loss': []}

for result in results:
    memory_size = result['memory_size']
    final_loss = result['smoothed_training_curve'][-1]
    data['memory_size'].append(memory_size)
    data['loss'].append(final_loss)

# Convert the data to a pandas DataFrame for easier plotting
df = pd.DataFrame(data)

# Create the swarmplot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='memory_size', y='loss', data=df)

# Add a light grid in the background
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Add labels
plt.xlabel('Memory size')
plt.ylabel('Loss')
plt.title('Loss per Memory Size Across Trials')

# Save the plot
plt.savefig('loss_vs_memory_size_swarmplot.png')


# create the training curve plots - one for each memory size, showing the unsmoothed loss as faint lines in the background, and the average smoothed loss as a bold line
# create a separate plot for each memory size
# the _average_ smoothed loss is the bold line - meaning we average the smoothed loss across all trials with the same memory size

smoothed_curves = []

for memory_size in df['memory_size'].unique():
    plt.figure(figsize=(10, 6))
    avg_smoothed_training_curve = torch.zeros(len(result['training_curve']))
    for result in results:
        if result['memory_size'] == memory_size:
            plt.plot(result['training_curve'], color='gray', alpha=0.1)
            avg_smoothed_training_curve += torch.tensor(result['training_curve'])


    avg_smoothed_training_curve /= len(df[df['memory_size'] == memory_size])
    smoothed_curves.append((memory_size, avg_smoothed_training_curve))
    plt.plot(avg_smoothed_training_curve, color='blue', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'Training Curve for Memory Size {memory_size}')
    plt.savefig(f'training_curve_memory_{memory_size}.png')

#sort the smoothed curves by memory size
smoothed_curves.sort(key=lambda x: x[0])

from scipy.ndimage import gaussian_filter1d
# create a plot showing the average smoothed loss for each memory size
plt.figure(figsize=(10, 6))
for memory_size, curve in smoothed_curves:
    # further smoothing for better visualization
    #use gaussian kernel for smoothing
    #dont use pandas, use scipy
    curve = gaussian_filter1d(curve, sigma=4)
    plt.plot(curve, label=f'Memory size {memory_size}')

plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Average Smoothed Training Curve for Different Memory Sizes')
plt.legend()

plt.savefig('average_smoothed_training_curve.png')

