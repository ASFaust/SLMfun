# Systematic RNN Dynamics Exploration
I am Exploring the forward dynamics of RNNS in a systematic way. 

## Execution Structure

1. `init_experiments.py` - Initializes the experiments by creating the file `rnn_configs.json` which contains the configurations of the RNNs to be explored.
If the file exists already, it only adds the configurations that are not already present in the file. 
It also creates the folder `results_01_all` which will contain the results of the first experiment.
2. `training_01_all.py` - Trains the RNNs with the configurations in `rnn_configs.json` and saves the results in `results_01_all`.
This script can be executed multiple times to continue training the RNNs. 
It can also be executed in parallel to speed up the training process if enough resources are available. 
I am using 2 parallel processes on my machine which has a RTX 3090 GPU. 
3. `examine_results_01_all.py` - Examines the results of the first experiment and creates the files
   * `results_01_all.png` - A plot of all training runs
   * `results_01_all_hyperparameters.png` - scatterplot of the influence of the hyperparameters on the results
   * `filtered_results_01_loss_below_1.9.png` - A plot of the hyperparameter influences of the RNNs that reached a loss below 1.9
     (a loss below 1.9 indicates successful usage of the memory of the RNN)

__These steps take about 3 hours on my machine.__

## Results

So far it seems that the best results are achieved with 
```commandline
    "use_input_gate": false,
    "use_forget_gate": true,
    "use_output_gate": false,
    "use_state_gate": true,
    "use_new_state": true,
    "use_old_state": false,
    "use_residual": true,
    "n_layers": 2,
    "state_size": 256,
```

Which is a 2 layer RNN with 256 units per layer, using the forget gate, a standard gated state update, and residual connections.
But 3 Layer RNNs are close behind. 
It is surprising that the forget gate is beneficial, while not using gating mechanisms for the input and output is beneficial.
I thought those would be beneficial as well.
That the 2 layer run is better than the 3 layer run is probably due to the more difficult nature of training a 3 layer RNN, which is a deeper network
with more parameters.
Also really beautiful to see is that the usage of new_state significantly improves the performance of the RNNs compared to only using the old state.
This indicates that the network learns temporal dependencies and uses them to predict the next step, even though we are not using any BPTT
and just alter the new state update based on the loss of the prediction of the next step. This is a very interesting result.

But this is with like 200 training runs remaining. 

## Next Steps In this project

1. Train the remaining RNNs
2. Analyze the results
3. Train the best RNNs for longer and with bigger state sizes
4. Evaluate on test data
