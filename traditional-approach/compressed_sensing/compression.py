import numpy as np
from scipy.fftpack import rfft, dct    
    
class TransformBasis:
    def _fft(self, N):    
        return rfft(np.identity(N))

    def _sparse(self, N):
        return np.identity(N)

    def _dct(self, N):
        return dct(np.identity(N))