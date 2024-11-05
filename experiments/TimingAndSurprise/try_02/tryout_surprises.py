import numpy as np
import matplotlib.pyplot as plt

# Define a range of probability values for predictions (correct class)
p_values = np.linspace(0.01, 0.99, 100)  # Avoid 0 and 1 to prevent log(0) issues

# True class label as one-hot vector
y_true = 1  # Index of the correct class in a one-hot vector

# Calculate each loss function
# Cross-Entropy Loss with epsilon for stability
epsilon = 1e-8
cross_entropy_loss = -np.log(p_values + epsilon)

# Absolute Difference Loss
abs_difference_loss = np.abs(p_values - 1)  # Since y=1 for the correct class

# Simple Difference Loss
difference_loss = p_values - 1

difference_loss *= difference_loss

# Normalize the losses to the range [0, 1] for fair comparison
def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

cross_entropy_loss = normalize(cross_entropy_loss)
abs_difference_loss = normalize(abs_difference_loss)
difference_loss = normalize(difference_loss)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(p_values, cross_entropy_loss, label="Cross-Entropy Loss (normalized)")
plt.plot(p_values, abs_difference_loss, label="Absolute Difference Loss (normalized)")
plt.plot(p_values, difference_loss, label="Difference Loss (normalized)")
plt.xlabel("Predicted Probability for Correct Class")
plt.ylabel("Normalized Loss Value")
plt.title("Comparison of Different Loss Functions")
plt.legend()
plt.grid(True)
plt.show()
