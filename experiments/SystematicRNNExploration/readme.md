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
```json
    "use_input_gate": false,
    "use_forget_gate": true,
    "use_output_gate": false,
    "use_state_gate": true,
    "use_new_state": true,
    "use_old_state": false,
    "use_residual": true,
    "n_layers": 2,
    "state_size": 256,
    "min loss": 1.5927601136607992,
```

Which is a 2 layer RNN with 256 units per layer, using the forget gate, a standard gated state update, and residual connections.
But 3 Layer RNNs are close behind. 
So the forward function of this RNN is:
```python
def forward(self,x,state):
    cat_input = torch.cat((x,state),dim=1)
    new_state = torch.tanh(self.linear1(cat_input))
    gate = torch.sigmoid(self.linear2(cat_input))
    new_state = gate * new_state + (1-gate) * state
    forget_gate = torch.sigmoid(self.linear3(cat_input))
    new_state = forget_gate * new_state 
    
    return new_state
```

It is surprising that the forget gate is beneficial, while not using gating mechanisms for the input and output is beneficial.
I thought those would be beneficial as well. The Forget gate has a clear advantage according to the empirical results.
That the 2 layer run is better than the 3 layer run is probably due to the more difficult nature of training a 3 layer RNN, which is a deeper network
with more parameters.
Also really beautiful to see is that the usage of new_state significantly improves the performance of the RNNs compared to only using the old state.
This indicates that the network learns temporal dependencies and uses them to predict the next step, even though we are not using any BPTT
and just alter the new state update based on the loss of the prediction of the next step. This is a very interesting result.

## Step 2: refining the best RNNs

After training all sensible configurations and filling the `results_01_all` folder, I will now train the best RNNs:
* Longer (50000 steps instead of 10000)
* With bigger state sizes (512 states)
* With 3 layers if the layer size wasn't 3 already

To this end, a second configuration list is generated with the execution of the script `init_experiments_02.py`.
This generates the file `rnn_configs_02.json` which contains the configurations of the best RNNs from the first experiment.
The best 10% of the RNNs are selected based on the loss of the training run. The best RNNs are then added up to 3 times to the new
configuration list with the changes mentioned above (bigger state size, 3 layers if not already 3 layers).
There are only layers with state size 256 and 512 in the new configuration list.
This also creates the folder `results_02_best` which will contain the results of the second experiment.

Then, the script `training_02_big.py` is executed to train with the configurations in `rnn_configs_02.json` and save the results in `results_02_best`.

Finally, the script `examine_results_02_big.py` is executed to examine the results of the second experiment and create the files
   * `results_02_big.png` - A plot of all training runs
   * `results_02_big_hyperparameters.png` - scatterplot of the influence of the hyperparameters on the results

## Next Steps In this project
1. Finish training `training_02_big.py`
2. Evaluate the results of the second experiment
3. Evaluate on test data 
4. Evaluate on multiple runs - so far I only trained each RNN once :D
5. Evaluate on byte pair encoding data instead of the simple byte level autoregressive language modelling



