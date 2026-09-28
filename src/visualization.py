from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(losses, output_path):
    """Plot the loss value recorded during training."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_decision_boundary(network, X, y, output_path):
    """Plot the model's decision boundary for 2D input data."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h = 0.01

    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5

    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )

    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    predictions = network.predict(mesh_points)
    predictions = predictions.reshape(xx.shape)

    plt.figure(figsize=(10, 8))

    plt.contourf(
        xx,
        yy,
        predictions,
        levels=50,
        alpha=0.8,
    )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y.flatten(),
        s=120,
        edgecolors="black",
    )

    plt.colorbar()

    plt.title("XOR Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()