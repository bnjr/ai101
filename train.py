from neuron import neuron
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_normalize_data(training_file):
    # Load combined data, skipping the first two rows (comment and header)
    data = np.loadtxt(training_file, delimiter=',', skiprows=2)
    training_inputs = data[:, 0]  # All rows, first column
    training_labels = data[:, 1]  # All rows, second column

    scaler = MinMaxScaler()
    training_inputs_normalized = scaler.fit_transform(training_inputs.reshape(-1, 1))
    return training_inputs_normalized, training_labels, scaler

def train_neuron(training_file, epochs, learning_rate):
    # Initialize weights and bias to random values
    weight = np.random.randn()
    bias = np.random.randn()
    
    training_inputs_normalized, training_labels, scaler = load_and_normalize_data(training_file)

    for _ in range(epochs):
        for input, label in zip(training_inputs_normalized, training_labels):
            # Forward pass: Compute the neuron's output
            output = neuron(input, weight, bias)
            error = label - output

            # Update weights and bias
            weight += learning_rate * error * input
            bias += learning_rate * error

    return weight, bias, scaler
