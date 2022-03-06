# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 00:00:55 2021

@author: brown
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.fftpack import rfft, rfftfreq
from datetime import datetime
import CSufrgs as cs

plt.close('all')

n  = 3000
f_min = 0
f_max = 50
a_min = 50
a_max = 100

m = 50
K = 6

resize = 1

t = np.linspace(0,1,n)


x, xclean  = cs.signal.spikes(n, K)
# x, xclean = cs.signal.normal(n, std=0.001, K=1)
# x, xclean = cs.signal.saw(n, w=0.2, K=2)
# x, xclean = cs.signal.ecg(n)

# x[x<0.2*max(x)] = 0

# trunc = 800
# xclean = x[:trunc] = 0
# xclean = x[-trunc:] = 0

f = rfft(x)
f /= n/2

if resize:
    biggest = max(f)
    # pdb.set_trace()
    
    comp_dB = np.array([20 * np.log10(abs(f_comp) / abs(biggest)) for f_comp in f])
    
    index_dB = np.argsort(comp_dB)    
    threshed = len(comp_dB[comp_dB > -30])
    maxf = max(index_dB[-threshed:])
    
    maxf = int(maxf/2)
    print(maxf)
# pdb.set_trace()

# x+= x2
# xclean += xclean2
# x, xclean = cs.signal.chirper(n, f_mi   n=f_min, f_max=50)
# x, xclean = cs.signal.ecg(n)
# x, xclean = cs.signal.emg(n)
# 

compression = cs.compress(x, m, basis='sparse', measurement='bernoulli')
rec = cs.recon(compression, 'omp2', t_stomp=2.2, K=K, tol=1e-12)
cs.metrics(compression, rec)


# xlong *= n/2

# max_s = np.max(abs(f))
# ts = 0.005*max_s
# f[np.where(abs(f) < ts)] = 0

# print(np.where(f == 0))

ffreq = rfftfreq(n, 1/n)

# plt.figure(1)
# plt.title("Frequency domain")
# plt.plot(ffreq, f, 'b', lw=0.5, label='Original signal')   # Plot of the transformed original signal
# plt.plot(ffreq, rec.s, 'r', lw=1, label = 'Sparse recontructed signal')       # Plot of reconstructed signal before inverse transform
# # plt.xlim((0, f_max*5))
# plt.legend()

plt.figure(2)
plt.title("Time domain")
plt.plot(t, x, 'b', lw=1.5, label='Original signal')     # Plot of the original signal x
plt.plot(t, rec.xrec, 'r', lw=1., label='CS reconstructed signal')      # Plot of reconstructed x
# plt.plot(t, compression.y, 'g.', lw=1., label='CS reconstructed signal') 
# plt.xlim((t[50], t[50]+50/f_max))
plt.legend()
plt.show()



