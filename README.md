# Digit Recognizer
An as-basic-as-it-gets node.js implementation of an artificial neural network for classifying handwritten digits. Configurable to explore how different network configurations and training parameters effect performance.

Training, testing, and gradient check modules expect the MNIST dataset files in a `mnist_dataset` directory at the project root (not included in repo; available at http://yann.lecun.com/exdb/mnist/). The server works 'out of the box'[^1] with the pre-trained params in sqlite files corresponding to a network configuration, where each number in the filename is a hidden layer with that number of neurons.

[^1]: assuming node.js and sqlite3 are installed, I'll probably add a docker setup at some point to standardize that
