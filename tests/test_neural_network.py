import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.xor import get_xor_data
from src.neural_network import NeuralNetwork
from src.training import train


def test_xor_data():
    X, y = get_xor_data()

    assert X.shape == (4, 2)
    assert y.shape == (4, 1)

    np.testing.assert_array_equal(
        X,
        np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=float),
    )

    np.testing.assert_array_equal(
        y,
        np.array([
            [0],
            [1],
            [1],
            [0],
        ], dtype=float),
    )


def test_forward_output_shape():
    np.random.seed(0)

    X, _ = get_xor_data()

    network = NeuralNetwork(
        layers=[2, 4, 1],
        learning_rate=0.5,
    )

    predictions = network.predict(X)

    assert predictions.shape == (4, 1)


def test_network_learns_xor():
    np.random.seed(0)

    X, y = get_xor_data()

    network = NeuralNetwork(
        layers=[2, 4, 1],
        learning_rate=0.5,
    )

    losses = train(
        network,
        X,
        y,
        epochs=4000,
        print_every=4000,
    )

    predictions = network.predict(X)

    binary_predictions = (
        predictions > 0.5
    ).astype(int)

    accuracy = np.mean(
        binary_predictions == y
    )

    assert losses[-1] < losses[0]
    assert accuracy == 1.0