import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from utils import isotropic_gaussian, load_iris_dataset


class EM:
    """
    Parameters:
        - K : number of gaussians
        - epsilon : stopping criterion precision
        - P : (K, ) probabilities p_k
        - r : (K, N) reponsibilities r_ki
        - mus : (K, D) Gaussian means 
        - Ds : (K, D) diagonals of Gaussian isotropic covariance matrices
        (where D is the dimension of the given samples)
    """
    def __init__(self, K, epsilon=1e-3, max_steps=50):
        self.K = K
        self.epsilon = epsilon
        self.P = None
        self.r = None
        self.mus = None
        self.Ds = None
        self.max_steps = max_steps

        # stored log likelihood values for stopping criterion
        self.nlls = []


    def _init_parameters(self, X):
        """Initialize EM parameters using k-means"""
        N, D = X.shape

        # initialize mu_k and p_k with k_means
        self.mus, labels, *_ = k_means(X, self.K)
        self.P = np.array([
            (labels == k).sum() / N for k in range(self.K)
        ])

        # set the D_k to the same value
        self.Ds = np.ones((self.K, D))


    def _e_step(self, x):
        """Expectation step""" 
        # Updating reponsabilities
        gaussians = np.array([
            isotropic_gaussian(
                x, 
                self.mus[k], 
                self.Ds[k]
            ) * self.P[k] for k in range(self.K)
        ])
        self.r = gaussians / gaussians.sum(0)

        # Storing the log likelihood
        nll = - np.log(gaussians.sum(0)).sum()
        self.nlls.append(nll)


    def _m_step(self, x):
        """Maximisation step."""
        N, D = x.shape
        Ns = self.r.sum(1)
        
        # update p_k
        self.P = Ns / N

        for k in range(self.K):
            # update mu_k
            self.mus[k] = x.T @ self.r[k] / Ns[k]

            # update D_k
            self.Ds[k] = np.sum(
                self.r[k, :, None] * (x - self.mus[k]) ** 2,
                axis=0
            ) / Ns[k]

    
    def fit(self, X):
        # initialization of the parameters
        self._init_parameters(X)

        # Main loop
        for _ in range(self.max_steps):
            # E step
            self._e_step(X)

            # M step
            self._m_step(X)

            # Stopping criterion
            if len(self.nlls) > 1 and np.abs(self.nlls[-1] - self.nlls[-2]) < self.epsilon:
                # Compute labels
                self.labels_ = self.r.max(axis=0)
                break


if __name__ == '__main__':
    # Load Iris dataset
    X, Z = load_iris_dataset()
    K = 4

    # Fit an EM model
    model = EM(K)
    model.fit(X)
    
    # Plot the nll
    plt.figure()
    plt.plot(
        model.nlls,
        ls='-',
        lw=1.5,
        marker='o',
        label='log likelihood'
    )
    plt.xlabel('iterations')
    plt.legend()
    plt.show()

    # Print the parameters
    print('p parameters:')
    print(model.P)
    print('\nmus:')
    print(model.mus)
    print('\nDs:')
    print(model.Ds)
    