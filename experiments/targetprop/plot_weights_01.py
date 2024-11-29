import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Create a directory to save the output images
os.makedirs('weights_images', exist_ok=True)

# Load the weights from the MNIST experiment
weights = np.load('linear1_w.npy')

print(weights.shape)  # Expected shape: (10, 785)

# Save a histogram of all weights
plt.hist(weights.flatten(), bins=100)
plt.title('Histogram of All Weights')
plt.xlabel('Weight Values')
plt.ylabel('Frequency')
plt.savefig('weights_images/weights_histogram.png')
plt.clf()

# Remove the bias weights
weights = weights[:, :-1]

n_filters = weights.shape[0]

# Reshape the weights to 28x28 for visualization
weights_reshaped = weights.reshape(n_filters, 28, 28)

# Save weight images for each neuron
for i in range(n_filters):
    plt.imshow(weights_reshaped[i], cmap='gray')
    plt.title(f'Weights for Neuron {i}')
    plt.colorbar(label='Weight Value')
    plt.savefig(f'weights_images/weight_{i:03d}.png')
    plt.clf()
