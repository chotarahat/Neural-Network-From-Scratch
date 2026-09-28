import sys
from pathlib import Path

import numpy as np

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from data.xor import get_xor_data
from src.neural_network import NeuralNetwork
from src.training import train


def main():
    np.random.seed(0)

    X, y = get_xor_data()

    network = NeuralNetwork(
        layers=[2, 4, 1],
        learning_rate=0.5,
    )

    print("Initial predictions:")
    print(network.predict(X))

    print()

    print("Training...")
    losses = train(
        network,
        X,
        y,
        epochs=4000,
        print_every=1000,
    )

    print()

    print("Final predictions:")
    print(network.predict(X))

    print()

    binary_predictions = (
        network.predict(X) > 0.5
    ).astype(int)

    accuracy = np.mean(
        binary_predictions == y
    )

    print("Binary predictions:")
    print(binary_predictions)

    print()

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()