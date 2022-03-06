# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 19:12:19 2021

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
import wfdb

sns.set()

#%% Import MIT database

mit_dir = 'C:/Users/brown/OneDrive/IC/MIT_arrhythmia_db/mit-bih-arrhythmia-database-1.0.0'

# test_files = [100, 101, 102, 107, 109, \
#               111, 115, 117, 118, 119]
test_files = [115]
test_records = list()
for file in test_files:
    test_record = wfdb.rdsamp(os.path.join(mit_dir, str(file)))
    test_record = test_record[0]
    test_record = np.array(test_record)[:,0]

    test_records.append(test_record)

test_records = np.ravel(np.array(test_records))

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
model = models.load_model('saved_models/CSNET_ecg_32.h5')

model.evaluate(r, test_x)

res = model.predict(r)[:,:,0]

test_x = test_x

#%% Plotting

seed = 1999

plt.plot(test_x[seed,:], label='Original data')
plt.plot(res[seed,:], label='Reconstructed using CSNet')
plt.legend()
plt.show()



