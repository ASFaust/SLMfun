i have an idea for a target propagation algorithm to directly obtain weights for a neural network that produce the correct output on a given input, target pair. my hope is that this leads to more interpretable, optimizer-free training. the only thing i dont have is a convergence guarantee, so i need to implement it in order to test it. 

the basic idea is the following: 

we distinguish between linear layers whose input can be changed and those whose input cannot be changed. the only layer whose input cannot be changed is the first layer that gets the input data as input. all others, the intermediate and output layer, can send change signals back to previous parts of the network.

given a training sample and a labeled output, my target propagation method efficiently computes a new set of parameters for the network, such that it outputs the labeled output given the training sample. my idea is to use the average of these newly computed parameters over a batch of samples to iteratively refine the weights of the network. as i said before, i have no convergence guarantees, so i need to test it.

we first concern ourselves with a single neuron from a linear layer. the bias is modeled as a constant "1" input, so there is no explicit bias handling, it's just an extra weight that gets adapted per neuron.

the input to this neuron is "x", its weights are "w", and its output is "v". the forward pass can be modeled as <x,w> = v.
the label, or target, for the output, is v', and based on that, we need to compute updates for both x and w. we denote those with x' and w'. they need to fulfill <x',w'> = v'.
we denote changes in those variables with the prefix "d": dv = v' - v. etc.
A very simple and promisig update is to use the same update into w that gradient descent would induce:

dw = x * dv = x * (v' - v) 

We could also attenuate that update in order to not change the weights that much:

dw = a * x * dv, with 0 < a < 1.

then, since we want to enforce <x',w'> = v', it follows that

dx = dv(1-a*|x|²)/|w + a * dv * x|² * (w + a * dv * x)

Computing those dx independently per-neuron in a layer is not sufficient though. since many neurons from the same layer share the same input values x but have different weights w, we will end up computing potentially many different target values x' for x.

To mitigate that, we could try averaging those x' first. This bears 2 problems:

* x' might lie outside the image of the activation function of the previous layer. e.g. x' = -1 with a relu layer as activation function of the previous layer. this target would not be possible to generate

* averaging the targets x' means that the newly computed parameters of the network as a whole will not lead to the correct computation of the labeled output. 
 
in order to mitigate these problems, we can do a couple more computation steps on the computed average intermediate target x', which i will from now on call ix. the goal of those transformations is to obtain a possible and sensible target value x', which can then be used to calculate the true weight update, such that <x',w'> = v'.

* first, we need to map ix to the activation function image. the way this is done is dependent on the activation function and might also be a good place to do certain kinds of regularizations. in our case, let's go with a relu activation: for that, we simply zero-out all negative entries of ix to obtain x'.

* we could then cap out too large values for numerical stability. we could say that we want our biggest value to be 2.0 for example, and set the target x' for any x greater than 2 to that value. this regularizes the intermediate activations to some extent.

This already gives us a new, sensible, realistic target for the input x', which can be further propagated back through the relu activation, and "forward propagated" in order to obtain the true new weight values w':

dw = (dv - <w,x'> + <w,x>) / |x'|² * x'

This finally gives us an update for the weights, considering a fixed target x' across all neurons of a layer.

then, we need to define how to propagate the target through a relu activation. For this, we need the input to the relu (v), the output (x), and its target output (x'). our goal is to compute a target for v, which we will call v', and which can be used as the target value for the linear layer that uses that relu activation. 
The way we do that is the following:
* if i <= 0 and x' == 0 (x >= 0 due to our alterations on it), then i' = i. we dont need to change anything here.
* if x' >= 0, then i' = x'. we are on the linear part of the relu, which feeds the learning signal right through.
* if x' == 0 and i >= 0, then i' = 0. we just push them to 0 if they're greater than 0 but should be 0. we could also use a negative constant here instead of 0 in order to push them past the boundary point, which might make things more stable.

then, all that is left is to concern ourselves with the last layer, which is the layer that receives the input. for this one, we have dx = 0 <=> x' = x, since we cannot change the input to that layer. this yields an update rule for its weights:

dw = dv/|x|² * x.

This concludes the computation steps for my target propagation idea. The hope is that with a small learning rate a, we obtain similar weights w' across a batch. the only problem is that the input layer does not have any such "small update" mechanisms. but this is left to be seen wether this might not just eventually also converge. maybe such a learning rate a is not needed at all and can be set to 1, but we need to investigate that computationally.