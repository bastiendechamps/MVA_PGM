import numpy as np
from sklearn.datasets import load_iris


def load_iris_dataset():
    """Returns the iris dataset using sklearn."""
    data = load_iris()
    X = data.data
    Z = data.target
    
    return X, Z


def isotropic_gaussian(x, mu, D):
    """Returns the value of the Gaussian desity with isotropic covariance.
    Args:
        - x : (N, d) data samples
        - mu : (d, ) Gaussian mean
        - D : (d, ) diagonal of isotropic covariance
    
    Returns:
        - density : (N, ) density values for each data sample
    """
    temp = -1 / 2. * ((x - mu) ** 2 / D).sum(1)
    density = np.exp(temp) / np.sqrt(2 * np.pi ** len(mu) * np.prod(D))

    return density