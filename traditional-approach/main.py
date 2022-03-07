#%% Imports

import matplotlib.pyplot as plt
from signals import BaseSignal
from compressed_sensing import CompressedSensing
from reconstruction import Reconstruction 
import pdb

#%% Initialization

N = 2048
M = 64
K = 20

maxFreq = 500
minFreq = 1
maxAmp = 50

#%% Signal

sg = BaseSignal(N, K, maxFreq, minFreq, maxAmp)
x = sg.addSignal('sinusoidal')


#%% Compressed Sensing

cs = CompressedSensing(x, M)

randomMatrix = 'bernoulli'
phi = cs.sampling(randomMatrix)

basis = 'fft'
psi = cs.compression(basis)

y = phi @ x
theta = phi @ psi

#%% Reconstruction

r = Reconstruction(theta, y, x, basis)
s, xrec = r.reconstruct('omp')

#%% Plotting

plt.figure(2)
plt.title("Time domain")
plt.plot(sg.t, x, 'b', lw=1.5, label='Original signal')     # Plot of the original signal x
plt.plot(sg.t, xrec, 'r', lw=1., label='CS reconstructed signal')      # Plot of reconstructed x
# plt.plot(t, compression.y, 'g.', lw=1., label='CS reconstructed signal') 
# plt.xlim((t[50], t[50]+50/f_max))
plt.legend()
plt.show()
