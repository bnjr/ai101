import numpy as np
from neuron import neuron
from train import train_neuron
from sklearn.preprocessing import MinMaxScaler

LEARNING_RATE = 0.01
EPOCHS = 1000

def read_evaluating_number(training_file):
    with open(training_file, 'r') as file:
        first_line = file.readline()
        evaluating_number = int(first_line.split()[-1])
    return evaluating_number

def interpret_output(output, evaluating_number, threshold=0.5):
    interpreted_outputs = []
    for value in output:
        probability = value * 100  # Convert to percentage
        if value > threshold:
            interpretation = f"Greater than {evaluating_number} ({probability:.2f}% probability)"
        else:
            interpretation = f"Less than {evaluating_number} ({probability:.2f}% probability)"
        interpreted_outputs.append(interpretation)
    return interpreted_outputs

def test_neuron(test_inputs, weight, bias, scaler, evaluating_number):
    normalized_test_inputs = scaler.transform(test_inputs.reshape(-1, 1))
    outputs = [neuron(input, weight, bias) for input in normalized_test_inputs]
    interpreted_outputs = [interpret_output(output, evaluating_number) for output in outputs]
    return interpreted_outputs

def main():
    print("Welcome to the Neuron Tester!")
    
    training_file = input("Enter the path to the training data file: ")
    weight, bias, scaler = train_neuron(training_file, EPOCHS, LEARNING_RATE)

    while True:
        try:
            test_input_str = input("\nEnter test inputs separated by space (or press Ctrl+C to exit): ")
            test_inputs = np.array(list(map(int, test_input_str.split())))
            test_outputs = test_neuron(test_inputs, weight, bias, scaler, read_evaluating_number(training_file))

            for input_value, output in zip(test_inputs, test_outputs):
                print(f"Test result for input {input_value}: {output}")

        except ValueError:
            print("Invalid input. Please enter integers only.")
        except KeyboardInterrupt:
            print("\nExiting the program.")
            break

if __name__ == "__main__":
    main()
