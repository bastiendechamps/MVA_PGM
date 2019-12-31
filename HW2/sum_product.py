import numpy as np
from scipy.special import logsumexp
import scipy.stats as stats
import matplotlib.pyplot as plt


class SumProductUndirectedChain:
    def __init__(self, psis_single_log, psis_double_log):
        """Initialize the undirected chain.
        Args:
            - psis_single_log: list of N np.array (node potential)
            - psis_double_log: list of N-1 np.array (edge potential)
        """
        self.psis_single_log = psis_single_log
        self.psis_double_log = psis_double_log
        self.mu_asc = []
        self.mu_desc = []
        self._Z = None
        self.propagated = False

    def __len__(self):
        return len(self.psis_single_log)

    def propagate(self):
        # Ascending messages
        mu_asc = 0
        self.mu_asc.append(np.zeros_like(self.psis_single_log[0]))
        for i in range(len(self) - 1):
            msg_single = mu_asc + self.psis_single_log[i]
            msg_double = self.psis_double_log[i].T
            mu_asc = logsumexp(msg_single + msg_double, axis=1)
            self.mu_asc.append(mu_asc)

        # Descending messages
        mu_desc = 0
        self.mu_desc.append(np.zeros_like(self.psis_single_log[-1]))
        for i in range(len(self) - 1, 0, -1):
            msg_single = mu_desc + self.psis_single_log[i]
            msg_double = self.psis_double_log[i - 1]
            mu_desc = logsumexp(msg_single + msg_double, axis=1)
            self.mu_desc.append(mu_desc)
        self.mu_desc.reverse()

        self.propagated = True

    def marginalize(self, idx):
        assert self.propagated, "Call propagate before marginalizing"
        # Compute un-normalized log probs
        log_probs = self.psis_single_log[idx] + self.mu_asc[idx] + self.mu_desc[idx]

        # Get back from log scale
        probs = np.exp(log_probs)

        # Normalize
        self._Z = np.sum(probs)
        probs /= self._Z

        return probs

    @property
    def Z(self):
        assert self.propagated, "Call propagate before normalization"
        if self._Z is None:
            self.marginalize(0)
        return self._Z


if __name__ == "__main__":
    # Test the sum product algorithm on independant Bernouilli distributions
    # Dimensions of each distribution
    n1, n2, n3 = 2, 5, 20

    # Bernouilli (unormalized)
    psi1 = np.array([4.0, 1.0])

    # Binomial (unormalized)
    psi2 = 5.0 * stats.binom.pmf(np.arange(n2), 5, 0.4)

    # Poisson (unormalized)
    psi3 = 4.0 * stats.poisson.pmf(np.arange(n3), 10.0)

    psis_single = [psi1, psi2, psi3]

    # No iteraction
    psis_double = [
        np.ones((n1, n2)),
        np.ones((n2, n3)),
    ]

    psis_single_log = [np.log(p) for p in psis_single]
    psis_double_log = [np.log(p) for p in psis_double]

    chain = SumProductUndirectedChain(psis_single_log, psis_double_log)
    chain.propagate()
    chain.marginalize(0)

    print("Partition function: Z =", chain.Z)

    # Some plots
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    for i in range(len(chain)):
        p = chain.marginalize(i)
        x = np.arange(len(p))
        axes[i].bar(x, p)
        axes[i].set_title("$p(x_{})$".format(i + 1))

    plt.show()
