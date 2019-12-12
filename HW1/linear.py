import data_utils
import numpy as np
import math
import matplotlib.pyplot as plt


def normal_equation(X, Y):
    """Solve the normal equation

    Args:
        - X
        - Y
    Returns:
        - theta, (3,) float array
    """
    X_tilde = np.hstack((X, np.ones((X.shape[0], 1))))
    theta = np.linalg.inv(X_tilde.T @ X_tilde) @ (X_tilde.T @ Y)
    return theta.flatten()


def linear_predict(X, theta, thresh=0.5):
    X_tilde = np.hstack((X, np.ones((X.shape[0], 1))))

    Y_hat = (X_tilde @ theta)[:, None]

    # Prediction
    return (Y_hat > thresh).astype(np.int32)


def linear_regression_experiment(X_train, Y_train, X_test, Y_test, name=''):
    if name != '':
        print("dataset {}".format(name))

    theta_hat = normal_equation(X_train, Y_train)
    separation_line = tuple(theta_hat - [0, 0, 0.5])
    _, ax = data_utils.plot_data_separation(X_test, Y_test, separation_line)
    ax.set_title("linear regression" if name == '' else "linear regression for set {}".format(name))
    plt.show()

    # Display parameters
    print(f"W = {theta_hat[0]:.2f}, {theta_hat[1]:.2f}, b = {theta_hat[2]:.2f}")

    # Compute metrics
    Y_pred = linear_predict(X_train, theta_hat)
    acc = (Y_pred == Y_train).mean()
    print(f"Accuracy on train set: {acc:.3f}")
    Y_pred = linear_predict(X_test, theta_hat)
    acc = (Y_pred == Y_test).mean()
    print(f"Accuracy on test set: {acc:.3f}")
    print()