import matplotlib.pyplot as plt
import json
import os
import numpy as np
import seaborn as sns

# Load the results
results = []

folder = "results_01_all"

files = os.listdir(folder)

for file in files:
    with open(os.path.join(folder, file), 'r') as f:
        results.append(json.load(f))

# Plot the results
plt.figure(figsize=(10, 5))

best_min_loss = float('inf')
best_min_loss_result = None

cleaned_results = [r for r in results if 'loss' in r]

for result in cleaned_results:
    plt.plot(result['loss'])
    if result['min loss'] < best_min_loss:
        print(f"new best loss: {result['min loss']}")
        best_min_loss = result['min loss']
        best_min_loss_result = result

plt.xlabel('Epoch')
plt.ylabel('Loss')

print("Best result:")
print_result = best_min_loss_result.copy()
del print_result['loss']
print(json.dumps(print_result, indent=4))

#save the plot
plt.savefig('results_01_all.png')

#close the plot
plt.close()
#clear the figure
plt.clf()

hyperparameters = ['state_size', 'use_input_gate', 'use_forget_gate', 'use_output_gate', 'use_state_gate', 'use_new_state', 'use_old_state', 'use_residual', 'n_layers']

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Loop through each hyperparameter and plot with seaborn swarmplot
for i, hyperparameter in enumerate(hyperparameters):
    values = [r[hyperparameter] for r in cleaned_results]
    losses = [r['min loss'] for r in cleaned_results]

    # Create the swarm plot
    plt.subplot(3, 3, i + 1)
    sns.swarmplot(x=values, y=losses, size=4, alpha=0.7)

    # Get unique values and calculate mean loss for each unique value of the hyperparameter
    unique_values = sorted(set(values))
    mean_losses = [np.mean([loss for v, loss in zip(values, losses) if v == val]) for val in unique_values]

    # Use indices for x positions to match categorical plotting
    x_positions = range(len(unique_values))  # Categorical index positions
    plt.scatter(x_positions, mean_losses, color='red', s=50, label='Mean', zorder=3)

    # Ensure x-ticks and labels match categorical values
    plt.xticks(ticks=x_positions, labels=unique_values)

    plt.xlabel(hyperparameter)
    plt.ylabel('Final Loss')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig('results_01_all_hyperparameters.png')
print(f"we have {len(cleaned_results)} results so far")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


# Convert cleaned_results to DataFrame if it’s not already
df = pd.DataFrame(cleaned_results)

# Separate features (X) and target (y)
X = df[hyperparameters]
y = df['min loss']

# take the log of the loss to make it more normally distributed
y = np.log(y)


# normalize the features
X = (X - X.mean()) / X.std()
# normalize the target
y = (y - y.mean()) / y.std()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Display coefficients for interpretation
coef_df = pd.DataFrame({'Hyperparameter': X.columns, 'Coefficient': model.coef_})
print(coef_df)

# Convert cleaned_results to a DataFrame if it’s not already
df = pd.DataFrame(cleaned_results)

# Filter the data to include only rows where the loss is less than 1.9
filtered_df = df[df['min loss'] < 1.9]

# List of hyperparameters
hyperparameters = ['state_size', 'use_input_gate', 'use_forget_gate', 'use_output_gate',
                   'use_state_gate', 'use_new_state', 'use_old_state', 'use_residual', 'n_layers']

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Loop through each hyperparameter and plot with seaborn swarmplot for filtered data
for i, hyperparameter in enumerate(hyperparameters):
    values = filtered_df[hyperparameter]
    losses = filtered_df['min loss']

    plt.subplot(3, 3, i + 1)
    sns.swarmplot(x=values.astype(int) if values.dtype == 'bool' else values, y=losses, size=3, alpha=0.7)

    # Get unique values and calculate mean loss for each unique value of the hyperparameter
    unique_values = sorted(set(values))
    mean_losses = [np.mean([loss for v, loss in zip(values, losses) if v == val]) for val in unique_values]

    # Use indices for x positions to match categorical plotting
    x_positions = range(len(unique_values))  # Categorical index positions
    plt.scatter(x_positions, mean_losses, color='red', s=50, label='Mean', zorder=3)


    plt.xlabel(hyperparameter)
    plt.ylabel('Final Loss')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # Adding a light grid

plt.tight_layout()
plt.savefig('filtered_results_01_loss_below_1.9.png')