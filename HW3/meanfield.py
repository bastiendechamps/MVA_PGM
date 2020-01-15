import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils import truncated_gaussian


class MeanField:
    def __init__(self, X, y, tau=100):
        self.X = X
        self.y = y
        self.tau = tau
        self.N, self.p = X.shape

        # Covariance matrix
        self.sigma = np.linalg.inv(np.eye(self.p) / self.tau + self.X.T @ self.X)

        self.beta_mean = np.random.randn(self.p)
        self.z_mean = np.random.randn(self.N)

        self.betas = []
        self.zs = []

    def train(self, n_iter):
        for i in range(n_iter):
            # Update beta
            self.beta_mean = self.sigma @ self.X.T @ self.z_mean
            self.betas.append(self.beta_mean[:])

            # Update z
            temp = self.X @ self.beta_mean
            self.z_mean = temp + self.y * norm.pdf(temp) / norm.cdf(self.y * temp)
            self.zs.append(self.z_mean[:])

    def sample(self, n_sample):
        # Sample betas
        betas = np.random.multivariate_normal(self.beta_mean, self.sigma, n_sample)

        # Sample zs
        zs = truncated_gaussian(self.z_mean, self.y, n_sample)

        return betas, zs

    def predict(self, x, n_sample=1000):
        betas, _ = self.sample(n_sample)
        z = x @ betas.T + np.random.randn(x.shape[0], n_sample)

        return np.sign(z)
