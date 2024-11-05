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
plt.show()
