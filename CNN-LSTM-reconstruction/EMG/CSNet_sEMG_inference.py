# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 19:06:36 2021

@author: brown
"""

#%% Imports 

from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from scipy.misc import electrocardiogram
import os
import scipy.io
from sEMG_feature_extraction import FeatureExtraction

sns.set()

#%% Import sEMG database

S1_A1 = scipy.io.loadmat('sEMG/S1_A1.mat')

sEMG = S1_A1['emg']

test_records = sEMG[:,0:3]
test_records = np.ravel(np.transpose(test_records))

#%% Formatting data

test_size = test_records.shape[0]
N = 256
test_entries = np.floor(test_size / N).astype('int')

excess = test_size % (N * test_entries)

test_x = np.reshape(test_records[excess:], (test_entries, N))   
 
#%% Compressed sensing

with open('random_matrices/phi32.txt', 'r') as arq:
    phi = arq.readlines()
    
for i, line in enumerate(phi):
    phi[i] = int(line)
    
phi = np.reshape(np.array(phi), (-1,N))
# pdb.set_trace()
r = list()

for test_set in test_x:
    y = phi @ test_set

    y_hat = phi.T @ y
    
    r.append((y_hat + np.mean(y_hat)) / np.std(y_hat))
    
r = np.array(r)
r = np.reshape(r, (-1,256,1))

#%% CNN
# pdb.set_trace()
model = models.load_model('saved_models/beta_EMG32_256.h5')

model.evaluate(r, test_x)

res = model.predict(r)[:,:,0]

test_x = test_x

#%% Plotting

seed = 55

plt.plot(test_x[seed,:], label='Original data')
plt.plot(res[seed,:], label='Reconstructed using CSNet')
plt.legend()
plt.show()


