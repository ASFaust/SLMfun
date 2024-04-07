# Small Language Model fun

This repo contains a lot of my ideas regarding recurrent neural networks, especially training autoregressive language models without backpropagation through time (bptt). 

## List of ideas

### Stimulus Layer

A certain kind of layer whose internal dynamics force it to learn useful representations even without BPTT.

### StraightBytes

Instead of encoding the input in one-hot vectors, each byte is represented as an 8-dimensional vector. This is used in a simple dense feed forward network, which receives the n previous bytes as input and tries to predict the next byte. The network is trained with a simple cross-entropy loss.

### SuccessiveAutoencoder

A recurrent autoencoder which is trained to retain its last input as well as its previous hidden state. This enforced memory could be used to train a language model.


### AutomatonNN

a matrix represents a sequence of events. a neural network needs to find out at which point in the sequence we are at, depending on the last event and the last input. that determines the output. the output is a softmax weighted sum of the matrix rows, where the softmax is determined by the input and the last state. the matrix entries are learned fixed parameters.

Turned out to be _not_ better than a same sized RNN. but is a great ablation since we can vary the number of states independently of the number of outputs. turns out even with as few as 2 states, it can learn to use them to improve its performance by a significant margin. we could make an ablation that shows number of states vs. performance. but the best performance is really not that good, around 1.6 cross entropy.


### GridEvaluationRNN

Just let a lot of different configurations of standard RNNs run. Let each run for 10 minutes.
Here is a list of parameters that can be varied:
* number of recurrent layers: 1 - 3
* number of bptt steps: (1,2,4) (i bet 2 is best)
* wether the new computed state is used in the same step forward pass
* wether the output of the recurrent layers is state or linear(cat(state,input)) (no activation if output gates, silu else)
* wether to use gates in the output of the recurrent layers (it seemed beneficial in the past)
* recurrent dynamic: gated, straight tanh, doubly gated: independent gates for forgetting and updating
* number of recurrent states: 32, 256
* number of hidden states if output is linear combination of state and input: 32, 256

Track the following metrics and try to make conclusions using plots between these metrics:
* learning curve
* number of parameters
* final cross entropy
* number of recurrent states

there are only two logical scatterplots: final cross entropy vs number of parameters and final cross entropy vs number of recurrent states. then also plot the learning curve (in a log plot) for each model. 