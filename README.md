# NeuralNetworkTest
This repository represents a series of experiments in constructing and implementing neural networks I have engaged in
for educational purposes, primarily using the EMNIST Datasets digit subset (which will be referred to as MNIST in most
of this repository)

## Simple Network

NeuralNetworks.py and ActivationFunctions.py implement a simple MLP network with Stochastic Gradient Decent
using only Numpy, which is then used in the Simple Generated and Simple MNIST notebooks (Simple Benchmarking provides
benchmark data for the network using generated data)

Simple Generated classifies a generated dataset from SciKit Learn (the dataset.make_blobs function), with a network of 
32-16-16-64 and 10000 data points it achieved an accuracy of 1.01% after 100 epochs

Simple MNIST classifies MNIST, using a 784-16-10 network, and achieved an error of 15.64% after 50 epochs

The primary limitation is the poor optimisation of the model, a 16 neuron hidden layer was the highest the network
could reasonably work with using the EMNIST dataset size and even then only trained for 50 epochs. The network is 
single threaded and CPU only, which drastically increases training time on even modest datasets for deep learning

The network also lacks any logging of data or meaningful persistence, which causes issues as any test data requires
an inference pass to collect said data, and does significant amount of in place manipulation of objects, which makes
parallel processing infeasible. A lack of data manipulation tools also means that any parallel processing would 
require moving the network to a new thread or device every run, rather than batchwise processing, and so any parallel
processing would likely end up slower due to overhead.

As such I have moved over to using Torch for further experiments, unless the implementation at a low level would be
educationally useful.

## Torch Networks

The Torch MNIST MLP notebook creates a simple MLP network in PyTorch and classifies MNIST with it, as a baseline. The 
only meaningful changes are the addition of a dropout layer at 0.1 before the initial and hidden layer, to combat overfitting, 
switching from an MSE loss function as implemented in NeuralNetworks.py to a Cross Entropy Loss function which better 
fits the dataset, and adding momentum to the SGD. The network is still fairly small as a 784-256-64-10 network, 
but this is significantly larger than the largest possible network of the simple implementation. The model reached an 
error of 2.0575% after 100 epochs, a significant amount better than the simple network implementation .The primary 
issue with the model is that the low size causes GPU overhead to be high and as such utilisation to be low. 
The most logical step forward is to then implement the model much bigger.

Torch MNIST MLP Big does just that, using a 784-1024-1024-256-64-10 network, several orders of magnitude bigger than the 
previous one. In order to accommodate the size a batch normalisation process is used after every layer, but otherwise
the architecture remains unchanged. The increase in size didn't result in an increase in compute time (as we expected 
given the overhead of the smaller network), but did garner a significant increase in performance, an error of 0.760%
after 100 epochs.

Torch MNIST Data Augmentations then takes that same network and uses data augmentations to increase the training data
available to us, specifically adding gaussian noise and a random affine transform.
The transformations are disabled during evaluation. After 100 epochs the network has an error of 0.902%, 
slightly worse than the non-augmented data, but that is to be expected as it is handling a significantly wider pool of data 
and trying to find consistent patterns. 
After 400 Epochs however, the error of the Big MLP network was 0.633%, a number that stayed fairly stable past 200 Epochs, 
whereas the Augmented MLP had an error of 0.565%, showing improvement as time goes one, with an error of 0.525% after
450 Epochs

The final step to improving the network architecture was to move to a more complex CNN based approach. Convolutional networks,
due to their inherent locality are ideal for image processing. A large number of approaches were tried but eventually a 
8 layer CNN followed by 2 fully connected layers proved the best combination of complexity and accuracy, with a max pool 
every 2 CNN layers and a doubling of the feature depth, resulting in a 2x2 image with 128 features passing into the fully
connected layers. After 250 Epochs the CNN produces an error of 0.22%, twice as good as the MLP approach.

As a final experiment, I loaded in the EMNIST Letters dataset and ran an expanded CNN on that dataset, using 16 CNN layers
by adding in 2 more layers to each size step before max pooling. The results were less impressive, primarily I imagine due
to the complexity of the dataset, the NIST letters are fairly hard to read as a human and there are 37 classes of them
compared to just 10 numbers, but even so the network got a 4.46% error percentage after 250 Epochs