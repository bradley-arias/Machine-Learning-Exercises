import numpy as np
import numpy.linalg as LA
import pickle
from PIL import Image

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = pickle.load(open('x_train.p', 'rb'), encoding='latin1')
    y_train = pickle.load(open('y_train.p', 'rb'), encoding='latin1')
    x_test = pickle.load(open('x_test.p', 'rb'), encoding='latin1')
    y_test = pickle.load(open('y_test.p', 'rb'), encoding='latin1')
    return x_train, y_train, x_test, y_test

def visualize_data(images: np.ndarray, controls: np.ndarray) -> None:
    """
    Args:
        images (ndarray): image input array of size (n, 30, 30, 3).
        controls (ndarray): control label array of size (n, 3).
    """
    # Current images are in float32 format with values between 0.0 and 255.0
    # Just for the purposes of visualization, convert images to uint8
    images = images.astype(np.uint8)
    
    # Your code here!
    # ---------------
    image_idx = [0, 10, 20] # from 5a: "visualize the 0th, 10th, and 20th images"
    for i in image_idx:
        im = Image.fromarray(images[i])
        im.show()
        print("Control: ", controls[i])
    return None

def compute_data_matrix(images: np.ndarray, controls: np.ndarray, standardize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        images (ndarray): image input array of size (n, 30, 30, 3).
        controls (ndarray): control label array of size (n, 3).
        standardize (bool): boolean flag that specifies whether the images should be standardized or not

    Returns:
        X (ndarray): input array of size (n, 2700) where each row is the flattened image images[i]
        Y (ndarray): label array of size (n, 3) where row i corresponds to the control for X[i]
    """
    # Your code here!
    # ---------------
    temp = np.zeros((images.shape[0], 2700))
    for i in range(images.shape[0]):
        temp[i] = images[i].flatten() # flatten each image to a row vector of size 2700
    if standardize:
        temp = temp/255.0 * 2 - 1 # standardize to [-1, 1]
    return temp, controls

def ordinary_least_squares(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        Y (ndarray): label array of size (n, 3).

    Returns:
        pi (ndarray): learned policy of size (2700, 3)
    """ 
    # Your code here!
    # ---------------
    return (LA.inv(X.T @ X) @ X.T) @ Y

def ridge_regression(X: np.ndarray, Y: np.ndarray, lmbda: float) -> np.ndarray:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        Y (ndarray): label array of size (n, 3).
        lmbda (float): ridge regression regularization term

    Returns:
        pi (ndarray): learned policy of size (2700, 3)
    """
    # Your code here!
    # ---------------
    return (LA.inv(X.T @ X + lmbda * np.eye(X.shape[1])) @ X.T) @ Y

def measure_error(X: np.ndarray, Y: np.ndarray, pi: np.ndarray) -> float:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        Y (ndarray): label array of size (n, 3).
        pi (ndarray): learned policy of size (2700, 3)

    Returns:
        error (float): the mean Euclidean distance error across all n samples
    """
    # Your code here!
    # ---------------
    temp = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
        temp[i] = LA.norm(np.dot(X[i,:].T, pi) - Y[i, :].T)**2 # squared error for each sample
    return np.sum(temp) / X.shape[0]

def compute_condition_number(X: np.ndarray, lmbda: float) -> float:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        lmbda (float): ridge regression regularization term

    Returns:
        kappa (float): condition number of the input array with the given lambda
    """
    # Your code here!
    temp = np.linalg.eigvals(X.T @ X + lmbda * np.eye(X.shape[1]))
    kappa = np.max(np.abs(temp)) / np.min(np.abs(temp))
    return float(kappa)

if __name__ == '__main__':

    x_train, y_train, x_test, y_test = load_data()
    print("successfully loaded the training and testing data")

    LAMBDA = [0.1, 1.0, 10.0, 100.0, 1000.0]

    # Your code here!
    # ---------------

    print("5a: Visualize Data")
    # 5a: visualize the 0th, 10th, and 20th images in the training
    # set along with their corresponding control labels
    visualize_data(x_train, y_train)
    # ---------------

    print("5b: Ordinary Least Squares")
    # 5b: perform ordinary least squares to learn optimal policy and report
    # what happens when you do this and explain why.
    X_train, Y_train = compute_data_matrix(x_train, y_train)
    X_test, Y_test = compute_data_matrix(x_test, y_test)
    pi_ols = ordinary_least_squares(X_train, Y_train)
    # Explanation: The matrix is singular and not invertible. There isn't
    # enough data for the high dimensional image space, so many solutions
    # exist. We have 91 images but 2700 parameters, so you can fit an
    # arbitrary set of controls to the training data.
    # ---------------

    print("5c: Ridge Regression")
    # 5c: perform ridge regression with all regularization values of lambda
    # and measure the test error with the average squared Euclidean distance
    for v in LAMBDA:
        pi_ridge = ridge_regression(X_train, Y_train, v)
        err_train = measure_error(X_train, Y_train, pi_ridge)
        err_test = measure_error(X_test, Y_test, pi_ridge)
        print("Lambda: ", v)
        print("Test Error: ", err_test)
        print()
    # ---------------

    print("5d: Standardized Data")
    # 5d: repeat 5c with standardized data.
    for v in LAMBDA:
        X_train, Y_train = compute_data_matrix(x_train, y_train, standardize=True)
        pi_ridge = ridge_regression(X_train, Y_train, v)
        err_train = measure_error(X_train, Y_train, pi_ridge)
        err_test = measure_error(X_test, Y_test, pi_ridge)
        print("Standardized - Lambda: ", v)
        print("Test Error: ", err_test)
        print()
    # ---------------

    print("5e: Evaluate Policies with and without standarization")
    # 5e: evaluate both polaicies (with and without standardization) on new validation
    # data and report the averaged squared Euclidean loss and qualitatively explain
    # how changing LAMBDA affects the performance in terms of bias and variance.
    standardize = [False, True]
    for s in standardize:
        X_train, Y_train = compute_data_matrix(x_train, y_train, standardize=s)
        X_test, Y_test = compute_data_matrix(x_test, y_test)
        err_train = np.zeros((len(LAMBDA),2))
        err_test = np.zeros((len(LAMBDA),2))
        kappa = np.zeros((len(LAMBDA),2))
        for i,v in enumerate(LAMBDA):
            pi = ridge_regression(X_train, Y_train, v)
            err_train[i, int(s)] = measure_error(X_train, Y_train, pi)
            err_test[i, int(s)] = measure_error(X_test, Y_test, pi)
            kappa[i, int(s)] = compute_condition_number(X_train, v)
        print("Standardize: ", s)
        print("Train Error: \n", err_train)
    # Explanation: The result with standarization show that the state space has high
    # dimensionality and the policy can't generalize well, adding bias. Increasing
    # lambda can worsen performance.
    # ---------------

    # 5f: For lambda = 100, report the condition number with and without standarization.
    print("5f: Condition Numbers")
    for s in standardize:
        X_train, Y_train = compute_data_matrix(x_train, y_train, standardize=s)
        kappa = compute_condition_number(X_train, 100.0)
        print("Standardize: ", s)
        print("Condition Number: ", kappa)