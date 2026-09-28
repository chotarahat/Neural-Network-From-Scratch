import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.xor import get_xor_data
from src.neural_network import NeuralNetwork


def main():
    np.random.seed(0)

    X, y = get_xor_data()

    network = NeuralNetwork(
        layers=[2, 4, 1],
        learning_rate=0.5,
    )

    activations = network.forward(X)
    predictions = activations[-1]

    loss = network.compute_loss(predictions, y)

    print("XOR Input:")
    print(X)

    print("\nTarget:")
    print(y)

    print("\nInitial Predictions:")
    print(predictions)

    print(f"\nInitial Loss: {loss:.4f}")


if __name__ == "__main__":
    main()