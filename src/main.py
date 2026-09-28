import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.xor import get_xor_data
from src.neural_network import NeuralNetwork
from src.training import train
from src.visualization import (
    plot_decision_boundary,
    plot_training_loss,
)


def main():
    np.random.seed(0)

    X, y = get_xor_data()

    network = NeuralNetwork(
        layers=[2, 4, 1],
        learning_rate=0.5,
    )

    print("=== Neural Network From Scratch ===")
    print()
    print("XOR Input:")
    print(X)

    print()
    print("Target:")
    print(y)

    print()
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

    predictions = network.predict(X)

    binary_predictions = (
        predictions > 0.5
    ).astype(int)

    accuracy = np.mean(
        binary_predictions == y
    )

    final_loss = losses[-1]

    print()
    print("Final predictions:")
    print(predictions)

    print()
    print("Binary predictions:")
    print(binary_predictions)

    print()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Final loss: {final_loss:.4f}")

    # Test on new points.
    test_X = np.array(
        [
            [0.5, 0.5],
            [0.2, 0.8],
            [0.8, 0.2],
        ],
        dtype=float,
    )

    test_predictions = network.predict(test_X)

    print()
    print("Test predictions:")
    for point, prediction in zip(
        test_X,
        test_predictions.flatten(),
    ):
        print(
            f"{point} -> "
            f"{prediction:.4f}"
        )

    # Save visualizations.
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    plot_training_loss(
        losses,
        output_dir / "training_loss.png",
    )

    plot_decision_boundary(
        network,
        X,
        y,
        output_dir / "decision_boundary.png",
    )

    print()
    print("Saved:")
    print(output_dir / "training_loss.png")
    print(output_dir / "decision_boundary.png")


if __name__ == "__main__":
    main()