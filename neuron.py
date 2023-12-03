import numpy as np

# Sigmoid activation function squashes 
# input values to a range between 0 and 1.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Neuron computation
def neuron(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    return sigmoid(weighted_sum)
