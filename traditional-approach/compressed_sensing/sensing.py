import numpy as np
import scipy.linalg as lng


class RandomMatrices:
    def gaussian(self, N, M):
        return np.random.randn(M, N)

    def toeplitz(self, N, M):
        sample_indices = np.random.choice(2, N)*2 - 1
        toe = lng.toeplitz(sample_indices)
        random_indices = np.random.choice(N, M, replace=False)
        return toe[random_indices,:]

    def bernoulli(self, N, M):
        return np.random.choice(2, (M, N))*2 - 1

    def uniform(self, N, M):
        random_indices = np.random.choice(N, M, replace=False)
        phi = np.zeros((M, N))
        phi[range(M),random_indices] = 1
        return phi