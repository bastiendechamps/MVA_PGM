import numpy as np 
from scipy.special import logsumexp

class SumProductUndirectedChain:
    
    def __init__(self, psis_single, psis_double):
        self.psis_single = np.log(psis_single)
        self.psis_double = np.log(psis_double)
        self.mu_asc = []
        self.mu_desc = []

    def __len__(self):
        return len(self.psis_single)

    def propagate(self):
        # Ascending
        self.mu_asc.append(np.zeros_like(self.psis_single[0]))
        for idx in range(len(self)):
            # Vector of shape (nb of states of node idx,)
            msg_single = self.mu_asc[idx] + self.psis_single[idx]
            # Matrix of shape (nb of states of node idx+1, nb of states of node idx) 
            msg_double = self.psis_double[idx].T
            # Append final message of shape (nb of states of node idx+1,)
            self.mu_asc.append(logsumexp(msg_double + msg_single, axis=1))

        # Descending
        self.mu_desc.append(np.zeros_like(self.psis_single[-1]))
        for idx in reversed(range(len(self))):
            # Vector of shape (nb of states of node idx,)
            msg_single = self.mu_desc[len(self) - 1 - idx] + self.psis_single[idx]
            # Matrix of shape (nb of states of node idx+1, nb of states of node idx) 
            msg_double = self.psis_double[idx].T
            # Append final message of shape (nb of states of node idx+1,)
            self.mu_desc.append(logsumexp(msg_double + msg_single, axis=1))

        self.mu_desc = list(reversed(self.mu_desc))
       

    def marginalize(self, idx):
        # Compute un-normalized log probs
        log_probs = self.psis_single[idx] + self.mu_asc[idx] + self.mu_desc[idx]

        # Get back from log scale
        probs = np.exp(log_probs)

        # Normalize
        Z = np.sum(probs)
        probs /= Z
        
        return probs

if __name__ == "__main__":
    # Test the sum product algorithm on independant Bernouilli distributions

    psis_single = [np.array([0.3, 0.7]), np.array([0.5, 0.5]), np.array([0.25, 0.75])]
    psis_double = [np.ones((2, 2)) for i in range(3)]

    A = SumProductUndirectedChain(psis_single, psis_double)
    A.propagate()
    A.marginalize(0)