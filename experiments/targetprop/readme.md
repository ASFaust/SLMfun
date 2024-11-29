# Target propagation

This code presents an alternative to backpropagation, called target propagation. 
The main idea is to directly compute weights that produce the correct output for a given input-output pair.
Each part of the network, such as relu, bias, linear layer have a backwards function, which takes in a target value for their outputs,
and computes how its parameters and input should change to produce that given target value.
The formulas for the backwards functions are modeled after traditional backpropagation, to make convergence more likely.

## Taking a closer look at backpropagation through single linear neurons

It all started with looking closer how each step in a stochastic gradient descent changes the outputs of linear neurons.
Let w be the weight vector of a linear neuron, x be its input, and y be its output.
then we can write y = <w,x>. I will call a gradient or desired change in the output of the linear neuron as dy.

Backpropagation derives the changes for w and x from dy using the chain rule as:

dw = dy * x

dx = dy * w

I would have expected that the changes in <w,x> would be proportional to the changes in w and x, but that is not the case.

My expectation was that 

<w + dw, x + dx> = <w + dy * x, x + dy * w> = <w, x> + dy

But if we plug those changes into our linear neuron, we get:

<w + dw, x + dx> = <w + dy * x, x + dy * w> = <w, x> (1 + dy²) + dy * (||x||² + ||w||²)

This highlights that the changes induced in the output of a linear neuron by changes in its inputs as derived by backpropagation are not linear. 

Of course, the reality is even more complex - We have adaptive learning rates scaling dw, and dx is also not linearily changed in practice,
but that makes the point even more clear: The changes in the output of a linear neuron are not linear in the changes of its inputs, and are dependent on 
the norms of the inputs and weights. 

This might explain why normalization techniques, such as batch, weight or layer normalization are so important in deep learning.
They might reduce the off-diagonal terms in the Hessian of the loss function, as they set either |w| or |x| to 1, 
which simplifies the above equation, eliminating the dependencies on those norms.

This might make it easier for the optimizers, such as Adam, to converge, as they can only approximate the diagonal terms of the Hessian matrix.

This might also explain why decaying learning rates are needed in deep learning, as the learning rate has to be adapted to the local curvature of the loss function, 
which gets more non-linear with the progress of training.

This all highlights the fact that even the most simple linear neurons have a complex loss surface, which is why the gradient is really just a very local approximation of the loss surface.

## Target propagation

The idea of target propagation is to directly compute the changes in the weights and inputs of a neuron that are needed to produce a given target output.

We start with taking the standard backpropagation formula for the x of a linear neuron:

dx = dy * w

we then compute a mean input x' of the layer, which is the mean of x + dx over all neurons in the layer.

Then we need to create a feasible x, which we call x''. This x'' is based on x', but for example, for a preceding relu layer, 
we need to find a feasible x'' that is closest to x' but lies within the image of the relu function. in this case, 
x'' = relu(x').

Then we can compute the changes in the weights of the linear neuron as:

dw = (y' - <x'',w>) / ||x''||² * x''

where y' is the target output of the linear neuron.

This ensures that 

<x'',w + dw> = y'

This is the main idea of target propagation. We can now apply this idea to more complex networks, such as multi-layer perceptrons.

I also defined bias, input and a few activation function layers. 
The input layer is a bit special: its feasible target is always its input, and it is not feasible to alter it.

## code

run `test_mnist.py` to generate new weights for a 2-layer MLP for a single sample of the MNIST dataset. 
this is mainly for testing wether the target propagation correctly computes the weights for a given input-output pair.

run `train_mnist.py` to train a 2-layer MLP on the MNIST dataset using target propagation.

## results

The target propagation algorithm seems to work, but it does not converge as well as backpropagation. 
My current best is at 90.5% training accuracy on the MNIST dataset with batch size 64.

An interesting phenomenon is that with batch size=1, it still converges to about 85% training accuracy.
batch size = 1 basically means that at each step, the weights are changed so that the network produces the correct output for the current input, 
without regard of preserving any previously learned knowledge. it basically overfits to the current training sample. 
That it still gets to 85% accuracy is interesting, because it means it still converges and retains some classification ability.
this means that the target propagation moves towards regions in the parameter space that are robust against itself.
it reminds me of fixed point iterations.

With this low batch size, the network converges very fast to the 85% accuracy, but then it gets stuck there. 
way faster than with batch size 64 or backpropagation.

## shortcomings

* Training accuracy 
* dependency on weight initialization: i observed that the target propagation algorithm is sensitive to the initial norm of the weights. 
They best get initialized with a norm of 0.001. If you initialize them with a norm of 1, the network will not converge. 
This is interesting as both times, we produce solutions to the current input-output pair, 
but the network with the larger weights does not seem to be able to leverage the new weights in order to generalize to the whole dataset.
* many activation function backward functions are unclear. i think i got it down for relu now. and the onehot one seems to work as well. 


## future work

* Test out different backwards functions for the activation functions: get rid of the onehot function and replace it with a binary classifier function.
* Rethink the linear layer backwards function. maybe dx = w * dy is not the best choice. dw = (y' - <x'',w>) / ||x''||² * x'' seems odd, especially the division by ||x''||².
* -> Layer norm: normalize the input of the linear layer to have a norm of 1. idk. find a reasoning for that.
* Weight normalization: separate scaling and rotation. this might improve convergence. find a reasoning for that.
* Sticky weights: once a weight is changed, make it harder to change it immediately. but how to still produce valid target weight configs is above me right now.


