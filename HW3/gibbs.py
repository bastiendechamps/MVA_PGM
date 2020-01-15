import numpy as np
import matplotlib.pyplot as plt
from utils import truncated_gaussian


class GibbsSampling:
    def __init__(self, X, y, tau, burnin=100):
        self.X = X
        self.y = y
        self.N, self.p = X.shape
        self.tau = tau
        self.burnin = burnin

        # Compute parameters of the posterior laws
        self.sigma = np.linalg.inv(np.eye(self.p) / self.tau + self.X.T @ self.X)

        # Initialize varying parameters
        self.mu = np.zeros(self.p)
        self.beta = np.random.randn(self.p)
        self.z = np.random.randn(self.N)
        self.mu_temp = self.sigma @ self.X.T

        # Storing samples
        self.zs = []
        self.betas = []

        # burn-in phase
        for _ in range(self.burnin):
            self._update()

    def _update(self):
        # Sample beta | z
        self.beta = np.random.multivariate_normal(self.mu, self.sigma)
        mean_post = self.X @ self.beta

        # Sample z | beta, y
        self.z = truncated_gaussian(mean_post, self.y)

        # update mu
        self.mu = self.mu_temp @ self.z

    def sample(self, n_sample):
        self.zs, self.betas = [], []
        for _ in range(n_sample):
            # Sample
            self._update()

            # Store samples
            self.zs.append(self.z[:])
            self.betas.append(self.beta[:])

        return np.array(self.betas), np.array(self.zs)

    def predict(self, x):
        assert len(self.betas) > 0, "Sample first!"
        n_sample = len(self.betas)
        betas = np.array(self.betas)
        z = x @ betas.T + np.random.randn(x.shape[0], n_sample)

        return np.sign(z)
