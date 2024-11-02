# New Architecture ideas

This folder is for testing out new ideas for the forward pass of neural netowrks.

A proper process of refining and testing these ideas is developed in order to track experiments and results.

The class of ideas to be tested is limited by the constraint that the architecture needs to be able to be implemented 
as a layer with associated states and a reset() method with arbitrary input and output sizes.

This constraint enables the use of the architecture within the same training framework, which allows for rapid 
iteration of ideas and testing of their effectiveness without having to reimplement the training and evaluation
frameworks.

the folder structure is designed with the aforementioned constraint in mind, as well as the consideration that the architectures
will be iteratively refined and tested.