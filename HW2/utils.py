import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from sklearn.datasets import load_iris
from scipy.linalg import eigh


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
    temp = -1 / 2.0 * ((x - mu) ** 2 / D).sum(1)
    density = np.exp(temp) / np.sqrt(2 * np.pi ** len(mu) * np.prod(D))

    return density


def confidence_ellipse(mean, cov, ax, n_std=1.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Args:
        - ax : the axes object to draw the ellipse into.
        - n_std : The number of standard deviations to determine the ellipse's radiuses.
        - kwargs : matplotlib.patches.Patch properties
    """
    # Eigendecomposition of the covariance matrix
    sigma, u = eigh(cov)

    # Compute ellipse features
    size = n_std * np.sqrt(sigma)
    theta = np.degrees(np.arctan2(*u[:, 0][::-1]))

    ellipse = Ellipse(
        xy=mean, width=size[0] * 5, height=size[1] * 5, angle=theta, alpha=0.2, **kwargs
    )

    # Add the ellipse to the current plot
    ax.add_patch(ellipse)


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def custom_data(N, sigma1=0.1, sigma2=1.0):
    """Generates a custom dataset (cross-shaped) to compare the EM algorithms and the K-Means.
    Args:
        - N : number of data samples in each Gaussian
        - rotation : angle to rotate the Gaussians

    Returns:
        - X : (N, 2) dataset
        - Z : (N,) labels
    """
    X1 = np.hstack(
        [np.random.normal(0, sigma1, size=N), np.random.normal(0, sigma2, size=N)]
    )

    X2 = np.hstack(
        [np.random.normal(0, sigma2, size=N), np.random.normal(0, sigma1, size=N)]
    )

    X = np.vstack([X1, X2]).T
    Z = np.hstack([np.zeros(N, dtype=int), np.ones(N, dtype=int)])

    return X, Z
