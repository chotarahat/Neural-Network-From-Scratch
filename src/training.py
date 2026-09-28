def train(network, X, y, epochs=4000, print_every=1000):
    """
    Train the neural network.

    Returns:
        losses: loss value recorded at every epoch
    """

    losses = []

    for epoch in range(epochs):

        # 1. Forward pass
        activations = network.forward(X)

        predictions = activations[-1]

        # 2. Calculate loss
        loss = network.compute_loss(
            predictions,
            y
        )

        losses.append(loss)

        # 3. Backward pass
        dW, dB = network.backward(
            activations,
            y
        )

        # 4. Update weights
        network.update_weights(
            dW,
            dB
        )

        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Loss: {loss:.4f}"
            )

    return losses