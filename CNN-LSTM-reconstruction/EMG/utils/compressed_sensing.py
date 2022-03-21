import numpy as np
import os

class CS:
    RM_BASEPATH = '../utils/random_matrices'

    def randomMatrix(self, fileName, N):
        self.N = N
        phiPath = os.path.join(self.RM_BASEPATH, fileName)

        with open(phiPath, 'r') as arq:
            self.phi = arq.readlines()

        for i, line in enumerate(self.phi):
            self.phi[i] = int(line)
            
        self.phi = np.reshape(np.array(self.phi), (-1, self.N))
        return self.phi

    def compress(self, x):
        normalized = list()

        for singleX in x:
            y = self.phi @ singleX
            y_hat = self.phi.T @ y
            
            normalized.append((y_hat - np.mean(y_hat)) / np.std(y_hat))
    
        return np.reshape(np.array(normalized), (-1, self.N, 1))   