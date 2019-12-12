import data_utils
import numpy as np
import matplotlib.pyplot as plt


def lda_mle_estimators(X, Y):
    """Return the MLE estimators for the LDA model

    Args:
        - X (N, 2)
        - Y (N, 1)
    Returns:
        - pi_mle
        - mu0_mle
        - mu1_mle
        - sigma_mle
    """
    Y0 = (Y == 0)[:, 0]
    X0, X1 = X[Y0], X[~Y0]
    N0 = (1 - Y).sum()
    N1 = Y.sum()

    # Estimator of pi
    pi_mle = np.mean(Y)

    # Estimator of mu_0
    mu0_mle = X0.sum(0) / N0

    # Estimator of mu_1
    mu1_mle = X1.sum(0) / N1

    # Estimator of sigma
    cov0 = (X0 - mu0_mle).T @ (X0 - mu0_mle) / N0
    cov1 = (X1 - mu1_mle).T @ (X1 - mu1_mle) / N1
    sigma_mle = (N0 * cov0 + N1 * cov1) / (N0 + N1)
    
    return pi_mle, mu0_mle, mu1_mle, sigma_mle


def lda_separation(pi_mle, mu0_mle, mu1_mle, sigma_mle):
    # Compute the coefficient of the separation line (see q.2b)
    sigma_inv = np.linalg.inv(sigma_mle)
    a0, a1 = sigma_inv @ mu0_mle, sigma_inv @ mu1_mle
    w = a1 - a0
    b = np.vdot(mu0_mle, a0) / 2. - np.vdot(mu1_mle, a1) / 2. - np.log((1 - pi_mle) / pi_mle)
    return np.hstack((w, b))


def lda_predict(X, theta, thresh=0.5):
    sigmoid = np.vectorize(lambda z: 1 / (1 + np.exp(-z)))

    # X_tilde as defined in the pdf
    X_tilde = np.hstack((X, np.ones((X.shape[0], 1))))

    # Y_hat is p of Y given X
    Y_hat = sigmoid(X_tilde @ theta)[:, None]

    # Prediction
    return (Y_hat > thresh).astype(np.int32)


def lda_experiment(X_train, Y_train, X_test, Y_test, name=''):
    if name != '':
        print("dataset {}".format(name))

    # Compute the MLE estimators of the data
    pi_mle, mu0_mle, mu1_mle, sigma_mle = lda_mle_estimators(X_train, Y_train)
    
    # Compute the separation line coefficients
    theta = lda_separation(pi_mle, mu0_mle, mu1_mle, sigma_mle)

    # Compute metrics
    Y_pred = lda_predict(X_train, theta)
    acc = (Y_pred == Y_train).mean()
    print(f"Accuracy on train set: {acc:.3f}")
    Y_pred = lda_predict(X_test, theta)
    acc = (Y_pred == Y_test).mean()
    print(f"Accuracy on test set: {acc:.3f}")
    print()

    # Plot the separation line with the test set
    separation_line = tuple(theta)
    _, ax = data_utils.plot_data_separation(X_test, Y_test, separation_line)
    ax.set_title("LDA separation" if name == '' else "LDA separation for set {}".format(name))
    plt.show()

    
