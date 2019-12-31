import itertools
import numpy as np
import matplotlib.pyplot as plt

from sum_product import SumProductUndirectedChain


class IsingGrid:
    def __init__(self, w, h, alpha, beta):
        self.w = w
        self.h = h
        self.beta = beta
        self.alpha = alpha

    def junction_tree(self):
        x = np.array(list(itertools.product(*[[0, 1]] * self.w)))

        phi_simple_log = self.alpha * x.sum(1) + self.beta * (
            x[:, :-1] == x[:, 1:]
        ).sum(1)

        phi_double_log = np.array(
            [
                [self.beta * (x[i] == x[j]).sum() for j in range(2 ** w)]
                for i in range(2 ** w)
            ]
        )

        phi_simple_log_chain = np.array([phi_simple_log for _ in range(self.h)])
        phi_double_log_chain = np.array([phi_double_log for _ in range(self.h)])

        chain = SumProductUndirectedChain(phi_simple_log_chain, phi_double_log_chain)

        return chain


if __name__ == "__main__":
    w = 10
    h = 100

    alpha = 0.0

    betas = np.linspace(0.0, 2.0, 20)
    log_zs = []

    for i, beta in enumerate(betas):
        model = IsingGrid(w, h, alpha, beta)
        chain = model.junction_tree()
        chain.propagate()
        chain.marginalize(0)

        log_zs.append(chain.logZ)

    # Plot it
    plt.figure(figsize=(4, 4))
    plt.plot(betas, log_zs, lw=2.0)
    plt.title("Partition function $Z(0, \\beta)$")
    plt.xlabel("$\\beta$")
    plt.ylabel("$\log Z(0, \\beta)$")
    plt.grid("on")
    plt.show()
