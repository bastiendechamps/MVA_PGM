import numpy as np 
import matplotlib.pyplot as plt
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
    def __init__(self, K, epsilon=1e-3):
        self.K = K
        self.epsilon = epsilon
        self.P = None
        self.r = None
        self.mus = None
        self.Ds = None

        # stored log likelihood values for stopping criterion
        self.nlls = []


    def _init_parameters(self, X):
        """Random initialization of the EM parameters."""
        mean = X.mean(axis=0)
        sigma2 = X.var(axis=0)

        # sample mu_k randomly
        self.mus = np.random.multivariate_normal(
            mean=mean, 
            cov=sigma2 * np.eye(len(mean)),
            size=self.K
        )

        # set the D_k to the same value
        self.Ds = np.array([sigma2 for _ in range(self.K)])

        # set the probabilities p_k without any prior
        self.P = np.ones(self.K) / self.K


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
        N, d = x.shape
        Ns = self.r.sum(1)
        
        # update p_k
        self.P = Ns / N

        # update mu_k
        self.mus = ((self.r @ x).T / Ns).T

        # update D_k
        # TODO: Remove the for
        # TODO: check if K = N does not make the wrong operation
        self.Ds = np.array([
            (self.r[k] * (x - self.mus[k]).T ** 2).T.sum(0)
         for k in range(self.K)])

    
    def fit(self, X):
        # initialization of the parameters
        self._init_parameters(X)

        # Main loop
        stop = False

        while not stop:
            # E step
            self._e_step(X)

            # M step
            self._m_step(X)

            # Stopping criterion
            stop = len(self.nlls) > 1 and np.abs(self.nlls[-1] - self.nlls[-2]) < self.epsilon


if __name__ == '__main__':
    # Load Iris dataset
    X, Z = load_iris_dataset()
    K = 2

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
    plt.show()

    # Print the parameters
    print('p parameters:')
    print(model.P)
    print('\nmus:')
    print(model.mus)
    print('\nDs:')
    print(model.Ds)
    