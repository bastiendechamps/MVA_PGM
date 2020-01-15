import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split


def load_data():
    """Load the German credit dataset.
    Returns:
        - X : (1000, 25) normalized inputs
        - y : (1000,) outputs (-1 and 1)
    """
    # Load data
    data = pd.read_csv("german.data-numeric", sep="\s+", header=None).values
    X, y = np.array(data[:, :-1], dtype=float), data[:, -1]
    N = X.shape[0]

    # Normalize data
    X = (X - X.mean(0)) / X.std(0)
    y = 2 * y - 3

    # Add a constant column to X
    X = np.hstack([X, np.ones((N, 1))])

    return X, y


def truncated_gaussian(mean, y, size=None):
    """Sample z with a truncated Gaussian.
    Args:
        - mean : (N,) array of means
        - y : (N, ) outputs in {-1, 1}
    Returns:
        - z : (N, ) sample of truncated Gaussian
    """
    a, b = -mean, -mean
    a[y < 0] = -np.inf
    b[y > 0] = np.inf
    if size is None:
        z = truncnorm.rvs(a, b, loc=mean)
    else:
        z = truncnorm.rvs(a, b, loc=mean, size=(size, len(mean)))

    return z
