import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import numpy as np
import math


def get_line_points(coeff, x1lim, x2lim):
    """Compute two points on the line

    Args:
        - coeff: a float triplet (a, b, c) defining the line a x1 + b x2 + c = 0
        coeff must be different from (0, 0, 0) 
    Returns:
        - x1
        - x2
    """
    a, b, c = coeff
    if b != 0:
        line = lambda x: -a * x / b - c / b
        x1 = list(x1lim)
        x2 = list(map(line, x1lim))
    elif a != 0:
        line = lambda x: -b * x / a - c / a
        x2 = list(x2lim)
        x1 = list(map(line, x2))
    else:
        raise ValueError("Line coeff must be different from (0, 0, 0)")
    return x1, x2


def plot_data_separation(X, Y, coeff=(0, 0, 0)):
    """Plot the data as a point cloud (x1, x2) in R^2 and a separation line

    Args:
        - X
        - Y
        - coeff: a float triplet (a, b, c) defining the line a x1 + b x2 + c = 0
        if coeff = (0, 0, 0), the line is not shown

    Returns:
        - fig
        - ax
    """
    Y0 = (Y == 0)[:, 0]
    X0, X1 = X[Y0], X[~Y0]
    fig, ax = plt.subplots()
    ax.scatter(X0[:, 0], X0[:, 1], marker='o', s=20, label='0')
    ax.scatter(X1[:, 0], X1[:, 1], marker='x', s=20, label='1')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim([X[:, 0].min() - 2, X[:, 0].max() + 2])
    ax.set_ylim([X[:, 1].min() - 2, X[:, 1].max() + 2])

    if coeff != (0, 0, 0):
        x1, x2 = get_line_points(coeff, ax.get_xlim(), ax.get_ylim())
        plt.plot(x1, x2, color="black", linestyle="--", linewidth=1)
    return fig, ax
