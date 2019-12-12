import data_utils
import numpy as np
import math
import matplotlib.pyplot as plt


def logistic_predict(X, theta, thresh=0.5):
    sigmoid = np.vectorize(lambda z: 1 / (1 + np.exp(-z)))

    # X_tilde as defined in the pdf
    X_tilde = np.hstack((X, np.ones((X.shape[0], 1))))

    # Y_hat is p of Y given X
    Y_hat = sigmoid(X_tilde @ theta)[:, None]

    # Prediction
    return (Y_hat > thresh).astype(np.int32)


def logistic_oracle(X, Y, theta):
    """Return the likelihood value and gradient for logistic regression
    
    Args:
        - X (N, 2)
        - Y (N, 1)
        - theta : (3,)
    Returns:
        - value : float or np.nan
        - gradient : (3,) np.float32 
        - hessian: (3, 3) np.float32 
    """
    sigmoid = np.vectorize(lambda z: 1 / (1 + np.exp(-z)))

    # X_tilde as defined in the pdf
    X_tilde = np.hstack((X, np.ones((X.shape[0], 1))))

    # Y_hat is p of Y given X and theta
    Y_hat = sigmoid(X_tilde @ theta)[:, None]
    error = Y_hat - Y

    # Value, gradient and Hessian
    val = -(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)).mean().item()
    grad = (error * X_tilde).mean(axis=0)
    hess = (
        Y_hat[:, None]
        * (1 - Y_hat[:, None])
        * np.matmul(X_tilde[:, :, None], X_tilde[:, None, :])
    ).mean(axis=0)

    return val, grad, hess


def newton_method(X, Y, oracle, theta, epsilon=1e-5):
    """Implementation of the Newton's method for a function f

    Args:
        - theta : (p, 1) float np.ndarray, initial guess of the parameters
        - epsilon : float, the precision
        - oracle : a function of X, Y, and theta, that return the value, gradient and hessian of f

    Returns:
        - theta_hist : list of estimates of optimal theta
        - val_hist : list of value of the cost
    """
    converged = False
    theta_hist = []
    val_hist = []
    val = np.inf
    while not converged:
        last_val = val
        val, grad, hess = oracle(X, Y, theta)
        theta -= np.linalg.inv(hess) @ grad
        if np.abs(val - last_val) < epsilon:
            converged = True
        theta_hist.append(theta.copy())
        val_hist.append(val)
    return theta_hist, val_hist


def logistic_regression_experiment(
    X_train, Y_train, X_test, Y_test, name="", show_iterations=False, epsilon=1e-3
):
    if name != "":
        print("dataset {}".format(name))

    theta_0 = np.array([0, 0, 0], dtype=np.float64)
    # Note : epsilon lower than 1e-3 causes singularity issues
    theta_hist, val_hist = newton_method(
        X_train, Y_train, logistic_oracle, theta_0, epsilon
    )

    # Show loss curve
    plt.title(f"Newton's method : logistic regression with $\epsilon = {epsilon}$")
    plt.xlabel("Iterations")
    plt.ylabel("Cross-entropy (log10-scale)")
    plt.plot(1 + np.arange(len(theta_hist)), np.log10(val_hist))
    plt.show()

    # Show iterative improvements
    for step, (theta, val) in enumerate(zip(theta_hist, val_hist)):
        if show_iterations or step == len(val_hist) - 1:
            separation_line = tuple(theta - [0, 0, 0.5])
            _, ax = data_utils.plot_data_separation(X_test, Y_test, separation_line)
            title = f"Logistic regression, step {step + 1}, loss {val:3f}"
            if name != "":
                title += ", dataset {}".format(name)
            ax.set_title(title)
            plt.show()

    # Display parameters
    theta_hat = theta_hist[-1]
    print(f"W = {theta_hat[0]:.2f}, {theta_hat[1]:.2f}, b = {theta_hat[2]:.2f}")

    # Compute metrics
    Y_pred = logistic_predict(X_train, theta_hat)
    acc = (Y_pred == Y_train).mean()
    print(f"Accuracy on train set: {acc:.3f}")
    Y_pred = logistic_predict(X_test, theta_hat)
    acc = (Y_pred == Y_test).mean()
    print(f"Accuracy on test set: {acc:.3f}")
    print()
