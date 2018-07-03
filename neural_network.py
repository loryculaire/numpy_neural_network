import numpy as np


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        np.random.seed(1)

        # We model a 3 layer network, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 4 matrix, with values in the range -1 to 1
        # and mean 0 for the first layerself.
        # And 4 x 1 matrix for the second layer
        self.synaptic_weights_0 = 2 * np.random.random((3,4)) - 1
        self.synaptic_weights_1 = 2 * np.random.random((4,1)) - 1

    #sigmoid function
    def __sigmoid(self, x, deriv=False):
        if(deriv==True):
            return x*(1-x)

        return 1/(1+np.exp(-x))

    def __ReLU(self, x, deriv=False):
        if(deriv==True):
            return 1. * (x > 0)

        return x * (x > 0)


    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            layer0 = training_set_inputs
            [layer1, layer2] = self.think(training_set_inputs)
            output = layer2

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            output_layer_error = training_set_outputs - output

            if(iteration % 10000) == 0:
                print("Error:" + str(np.mean(np.abs(output_layer_error))))

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.

            #backpropagation
            layer_2_delta = output_layer_error * self.__sigmoid(output, deriv=True)
            layer_1_error = layer_2_delta.dot(self.synaptic_weights_1.T)
            layer_1_delta = layer_1_error * self.__sigmoid(layer1, deriv=True)

            #update weights - gradient descent
            self.synaptic_weights_1 += layer1.T.dot(layer_2_delta)
            self.synaptic_weights_0 += layer0.T.dot(layer_1_delta)

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network.
        layer1 = self.__sigmoid(np.dot(inputs, self.synaptic_weights_0))
        layer2 = self.__sigmoid(np.dot(layer1, self.synaptic_weights_1))

        return [layer1, layer2]


def run():
    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights_0)
    print(neural_network.synaptic_weights_1)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = np.array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights_0)
    print (neural_network.synaptic_weights_1)

    # Test the neural network with a new situation.
    print ("Considering new situation [1, 0, 0] -> ?: ")
    [layer1, layer2] = neural_network.think(np.array([1, 0, 0]))
    print(layer2)


if __name__ == '__main__':
    run()
