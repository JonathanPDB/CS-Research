
import numpy as np
from .inverse import InverseTransform
from .algorithms import ReconAlgs

class Reconstruction:
    def __init__(self, theta, y, x, basis, tol=1e-10):
        self.theta = theta
        self.y = y
        self.x = x
        self.tol = tol
        self.basis = basis
        self.N = x.shape[0]
        self.M = y.shape[0]

    def reconstruct(self, method, *args, **kwargs):
        self.s, self.i = getattr(ReconAlgs(), method)(self, *args, **kwargs)
        self.inverseTransform()
        return self.s, self.recon

    def thresholdOutput(self, threshold):
        max_s = np.max(abs(self.s))
        ts = threshold*max_s
        self.s[np.where(abs(self.s) < ts)] = 0

    def inverseTransform(self):
        self.recon = getattr(InverseTransform(), '_i' + self.basis)(self)

    def offset(self, value):
        self.recon += value

