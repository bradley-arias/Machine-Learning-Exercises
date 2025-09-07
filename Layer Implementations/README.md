# Layer Implementations

1. Implement the forward and backward passes of the fully-connected layer in `layers.py`. Do not iterate over training examples; use batched operations.

2. Implement the forward and backward passes of the softmax activation in `activations.py`. You my use a `for` loop over the training points in the mini-batch.

3. Implement the forward and backward passes of the cross-entropy cost in `losses.py`. Do not iterate over training examples; use batched operations.

4. Fill in the `forward`, `backward`, `predict` methods for the `NeuralNetwork` class in `models.py`. Define the parameters in `train_ffnn.py`.

5. Train a 2-layer neural network on the Iris Dataset by running `train_ffnn.py`.