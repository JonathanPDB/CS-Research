from .signal import Signal
import numpy as np

class BaseSignal:
    def __init__(self, N, K=1, maxFreq = 0, minFreq = 1, maxAmp = 0):
        self.N = N
        self.K = K
        self.maxFreq = maxFreq
        self.minFreq = minFreq
        self.maxAmp = maxAmp
        self.t = np.arange(0, 1, 1/self.N)
        self.x = np.zeros(self.N)

    def addSignal(self, type, *args, **kwargs):
        self.newX = getattr(Signal(), type)(self, *args, **kwargs)
        self.x += self.newX
        return self.x

    def addGaussianNoise(self, SNR):
        self.clean = self.x        
        N = self.x.shape[0]
    
        Px_dB = 10 * np.log10(np.mean(self.x**2))
            
        noise_dB = Px_dB - SNR
        noise_t = 10 ** (noise_dB / 10)
        
        noise = np.random.normal(0, np.sqrt(noise_t), N)
        self.x += noise

        return self.x



