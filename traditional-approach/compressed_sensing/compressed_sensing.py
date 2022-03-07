import numpy as np
from .sensing import *
from .compression import *


class CompressedSensing:
    def __init__(self, x, M):
        self.x = x
        self.N = x.shape[0]
        self.M = M

    def sampling(self, random_matrix, normalize = True):
        self.phi = getattr(RandomMatrices(), random_matrix)(self.N, self.M)
        return self.normalizeMatrix() if normalize else self.phi

    def compression(self, basis):
        self.psi = getattr(TransformBasis(), '_' + basis)(self.N)
        return self.psi

    def normalizeMatrix(self):
        rows    = np.split(self.phi, self.M)
        
        matrix = []
        for row in rows:
            norm = np.linalg.norm(row)
            if norm==0:
                matrix.append(row)
            else:
                matrix.append(np.squeeze(row/norm))
        
        self.phi = np.reshape(matrix, (self.M, self.N))
        return self.phi