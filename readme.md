# Small Language Model fun

This repo contains a lot of my ideas regarding recurrent neural networks, especially training autoregressive language models without Backpropagation through time. 

## List of ideas

### Stimulus Layer

A certain kind of layer whose internal dynamics force it to learn useful representations even without BPTT.

### StraightBytes

Instead of encoding the input in one-hot vectors, each byte is represented as an 8-dimensional vector. This is used in a simple dense feed forward network, which receives the n previous bytes as input and tries to predict the next byte. The network is trained with a simple cross-entropy loss.

### SuccessiveAutoencoder

A recurrent autoencoder which is trained to retain its last input as well as its previous hidden state. This enforced memory could be used to train a language model.