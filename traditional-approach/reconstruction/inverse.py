from scipy.fftpack import irfft, idct

class InverseTransform:
    def _ifft(self, r):
        xrec = irfft(r.s, r.N)
        xrec *= r.N/2  
        return xrec

    def _idct(self, r):
        xrec = idct(r.s)
        return xrec

