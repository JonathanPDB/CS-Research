import numpy as np
from scipy.stats import norm
from scipy.signal import sawtooth, chirp

class Signal:
    def sinusoidal(self, bs):
        x = bs.maxAmp * np.cos(2 * np.pi * bs.maxFreq * bs.t)
        return x

    def normal(self, bs, mean = 0.5, std = 0.01):
        meanK = mean/(bs.K+1)

        x = np.zeros(bs.N)
        
        for i in range(bs.K):
            xK = norm.pdf(bs.t, meanK, std) * bs.maxAmp
            x += xK
            
            meanK += mean
           
        return x

    def spikes(self, bs, minAmp = 0):
        x = np.zeros(bs.N)

        if bs.maxFreq == None:    # Avoid Aliasing
            bs.maxFreq = np.floor(bs.N/2 - 1)  
            
        peaks = np.random.randint(minAmp, bs.maxAmp, size = bs.K)
        freqs = np.random.randint(bs.minFreq, bs.maxFreq, size = bs.K)
        
        for i in range(freqs.shape[0]):
            if i % 2:
                x[freqs[i]] = peaks[i]
            else:
                x[freqs[i]] = -peaks[i]

        return x

    def saw(self, bs, width = 0.1):
        divPerK = int(np.floor(bs.N/bs.K))
        tK = np.arange(0, 1, 1/divPerK)
        
        sawK = sawtooth(2*np.pi*tK, width) * bs.maxAmp
        
        x = np.zeros(bs.N)
        
        for i in range(bs.K):
            x[i*divPerK:divPerK*(i+1)] += sawK
           
        diff = bs.N - divPerK*bs.K
        if diff > 0:
            x = x[:divPerK*bs.K]
            x = np.pad(x, (0, diff), 'constant', constant_values=sawK[-1])
            
        return x

    def chirper(self, bs):
        return chirp(bs.t, f0=bs.minFreq, f1=bs.maxFreq, t1=1, method='linear')

    