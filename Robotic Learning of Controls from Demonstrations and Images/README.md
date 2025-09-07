# Robotic Learning of Controls from Demonstrations and Images

For these tasks we were limited to only using `numpy` and `numpy.linalg` for performing any computations.

1. Visualize the `0th`, `10th`, and `20th` images in the training dataset and their corresponding control vectors.

2. Report what happens when you try to learn the optimal policy using the ordinary least squares solution.

3. Report the training error results for ridge regression with $\lambda = [0.1, 1.0, 10, 100, 1000]$.

4. Standardize the training data to be within the range [-1, 1]. Repeat the previous part and report the average squared training error for each value of $\lambda$.

5. Evaluate both policies (with and without standardization) on the new validation data `x_test.p` and `y_test.p` for different values of $\lambda$. Report the average squared Euclidean loss and qualitatively explain how changing the values of $\lambda$ affects the performance in terms of bias and variance.

6. Report the condition number with and without the standardization technique applied.