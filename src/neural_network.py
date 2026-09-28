import numpy as np


class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.5):
        """
        Initialize the neural network.

        Example:
            layers = [2, 4, 1]

        Meaning:
            2 input neurons
            4 hidden neurons
            1 output neuron
        """
        self.layers = layers
        self.learning_rate = learning_rate

        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            weights = (
                np.random.randn(layers[i], layers[i + 1])
                * np.sqrt(2.0 / layers[i])
            )

            biases = np.zeros((1, layers[i + 1]))

            self.weights.append(weights)
            self.biases.append(biases)

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    @staticmethod
    def sigmoid_derivative(a):
        """
        Derivative of sigmoid when a = sigmoid(z).
        Used in Part 2 during backpropagation.
        """
        return a * (1 - a)

    @staticmethod
    def tanh(x):
        """Tanh activation function."""
        return np.tanh(x)

    @staticmethod
    def relu(x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def forward(self, X):
        """
        Perform forward propagation.

        Returns:
            activations from input through output layer.
        """
        activations = [X]
        current = X

        for i in range(len(self.weights)):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            current = self.sigmoid(z)
            activations.append(current)

        return activations

    @staticmethod
    def compute_loss(y_pred, y_true):
        """Binary cross-entropy loss."""
        epsilon = 1e-15

        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.mean(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

        return loss

    def predict(self, X):
        """Generate predictions."""
        return self.forward(X)[-1]