import itertools

import numpy as np

from sum_product import SumProductUndirectedChain

w = 10
h = 100

alpha = 0
beta = 0.5

x = np.array(list(itertools.product(*[[0, 1]] * w)))
assert x.shape == (2 ** w, w)

phi_simple = np.exp(alpha * x.sum(1)) * np.exp(beta * (x[:, :-1] == x[:, 1:]).sum(1))
assert phi_simple.shape == (1024,)

# SLOW
phi_double = np.empty((2 ** w), (2**w))
for i in range(2 ** w):
    for j in range(2 ** w):
        phi_double[] # TODO


phi_double = np.exp(beta * (x[:-1, :] == x[1:, :]).sum(0)) # TODO
assert phi_double.shape == (1024, 1024)


# chain = SumProductUndirectedChain(phi_double, phi_double)

# assert len(chain) == 100

# chain.propagate()

# Z = chain.Z
