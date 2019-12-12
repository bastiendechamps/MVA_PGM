import data_utils
import numpy as np
import matplotlib.pyplot as plt


def qda_mle_estimators(X, Y):
    """Return the MLE estimators for the QDA model

    Args:
        - X (N, 2)
        - Y (N, 1)
    Returns:
        - pi_mle
        - mu0_mle
        - mu1_mle
        - sigma0_mle
        - sigma1_mle
    """
    Y0 = (Y == 0)[:, 0]
    X0, X1 = X[Y0], X[~Y0]
    N0 = (1 - Y).sum()
    N1 = Y.sum()

    # Estimator of pi
    pi_mle = N1 / (N1 + N0)

    # Estimator of mu_0
    mu0_mle = X0.sum(0) / N0

    # Estimator of mu_1
    mu1_mle = X1.sum(0) / N1

    # Estimator of sigma
    sigma0_mle = (X0 - mu0_mle).T @ (X0 - mu0_mle) / N0
    sigma1_mle = (X1 - mu1_mle).T @ (X1 - mu1_mle) / N1
    
    return pi_mle, mu0_mle, mu1_mle, sigma0_mle, sigma1_mle


def qda_ellipse_separation(pi_mle, mu0_mle, mu1_mle, sigma0_mle, sigma1_mle):
    # Compute the ellipse coefficients for the separation line
    sigma0_inv = np.linalg.inv(sigma0_mle)
    sigma1_inv = np.linalg.inv(sigma1_mle)
    a0, a1 = sigma0_inv @ mu0_mle, sigma1_inv @ mu1_mle

    # the ellipse equation is x^T A x + w^T x + b = 0
    A = sigma0_inv / 2. - sigma1_inv / 2.
    w = a1 - a0
    b = np.log(np.linalg.det(sigma0_mle) / np.linalg.det(sigma1_mle)) / 2. - np.log((1 - pi_mle) / pi_mle) + np.vdot(a0, mu0_mle) / 2. - np.vdot(a1, mu1_mle) / 2.

    return A, w, b


def qda_predict(X, A, w, b, thresh=0.5):
    sigmoid = np.vectorize(lambda z: 1 / (1 + np.exp(-z)))

    # Y_hat is the associated quadratic form
    Y_hat = (X * (X @ A)).sum(axis=1) + (X @ w.reshape((2, 1))).flatten() + b
    Y_hat = Y_hat.reshape((len(Y_hat), 1))

    # Prediction
    return (Y_hat > thresh).astype(np.int32)


def qda_plot_separation(X, Y, A, w, b, name, thresh=0.5):
    # Create a mesh
    x1 = np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 1000)
    x2 = np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 1000)
    mesh = np.meshgrid(x1, x2)
    mesh_values = qda_predict(np.array([mesh[0].flatten(), mesh[1].flatten()]).T, A, w, b, thresh)
    mesh_values = mesh_values.reshape((len(x1), len(x2)))

    # plot the separation
    Y0 = (Y == 0)[:, 0]
    X0, X1 = X[Y0], X[~Y0]
    fig, ax = plt.subplots()
    ax.scatter(X0[:, 0], X0[:, 1], marker='o', s=20, label='0')
    ax.scatter(X1[:, 0], X1[:, 1], marker='x', s=20, label='1')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim([X[:, 0].min() - 2, X[:, 0].max() + 2])
    ax.set_ylim([X[:, 1].min() - 2, X[:, 1].max() + 2])
    ax.contour(mesh[0], mesh[1], mesh_values, [0], colors="black", linestyles="--", linewidths=1)
    ax.set_title("QDA separation" if name == '' else "QDA separation for set {}".format(name))
    plt.legend()
    plt.show()
    

def qda_experiment(X_train, Y_train, X_test, Y_test, name=''):
    if name != '':
        print("dataset {}".format(name))

    # Compute the MLE estimators of the data
    pi_mle, mu0_mle, mu1_mle, sigma0_mle, sigma1_mle = qda_mle_estimators(X_train, Y_train)
    print('pi:', pi_mle)
    print('mu_0:', mu0_mle)
    print('mu_1:', mu1_mle)
    print('sigma_0:', sigma0_mle)
    print('sigma_1:', sigma1_mle)

    # Compute the separation line coefficients
    A, w, b = qda_ellipse_separation(pi_mle, mu0_mle, mu1_mle, sigma0_mle, sigma1_mle)
    
    # Compute metrics
    Y_pred = qda_predict(X_train, A, w, b)
    acc = (Y_pred == Y_train).mean()
    print(f"Accuracy on train set: {acc:.3f}")
    Y_pred = qda_predict(X_test, A, w, b)
    acc = (Y_pred == Y_test).mean()
    print(f"Accuracy on test set: {acc:.3f}")
    print()

    # Plot the separation line with the test set
    qda_plot_separation(X_test, Y_pred, A, w, b, name)
    