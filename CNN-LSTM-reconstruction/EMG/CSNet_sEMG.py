# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:35:03 2021

@author: brown
"""
#%% Imports

import numpy as np
# import pandas as pd
import pdb
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import pywt
from CSNet import CSNet

sns.set()


#%% sEMG database

S1_A1 = scipy.io.loadmat('sEMG/S1_A1.mat')
sEMG = S1_A1['emg']

test_records = sEMG[:,0:2]
test_records = np.transpose(test_records)

train_records = sEMG[:,2:8]                                                                                                                                                                                                                                
train_records = np.transpose(train_records)

#%% Divide in blocks

N = 256

train_x = list()
for row in train_records:
    row_size = row.shape[0]
    row_entries = np.floor(row_size / N).astype('int')

    excess = row_size % (N * row_entries)

    train_x.append(np.reshape(row[excess:], (row_entries, N)))

train_x = np.array(train_x)
train_x = np.reshape(train_x, (train_x.shape[0]*train_x.shape[1], train_x.shape[2]))


test_x = list()
for row in test_records:
    row_size = row.shape[0]
    row_entries = np.floor(row_size / N).astype('int')

    excess = row_size % (N * row_entries)

    test_x.append(np.reshape(row[excess:], (row_entries, N)))

test_x = np.array(test_x)
test_x = np.reshape(test_x, (test_x.shape[0]*test_x.shape[1], test_x.shape[2]))


#%% Wavelets

train_cA = list()
train_cD1 = list()
train_cD2 = list()
train_cD3 = list()

for row in train_x:
    cA, cD1, cD2, cD3 = pywt.wavedec(row, 'db5', level=3)
    train_cA.append(cA)
    train_cD1.append(cD1)
    train_cD2.append(cD2)
    train_cD3.append(cD3)

train_cA = np.array(train_cA)
train_cD1 = np.array(train_cD1)
train_cD2 = np.array(train_cD2)
train_cD3 = np.array(train_cD3)



test_cA = list()
test_cD1 = list()
test_cD2 = list()
test_cD3 = list()

for row in test_x:
    cA, cD1, cD2, cD3 = pywt.wavedec(row, 'db5', level=3)
    test_cA.append(cA)
    test_cD1.append(cD1)
    test_cD2.append(cD2)
    test_cD3.append(cD3)

test_cA = np.array(test_cA)
test_cD1 = np.array(test_cD1)
test_cD2 = np.array(test_cD2)
test_cD3 = np.array(test_cD3)


#%% CS

with open('random_matrices/phi16.txt', 'r') as arq:
    phi = arq.readlines()
    
for i, line in enumerate(phi):
    phi[i] = int(line)
    
phi = np.reshape(np.array(phi), (-1,N))


def sensing(data):
    r = list()
    size = data.shape[1]
    wt_phi = phi[:, :size]
    
    # pdb.set_trace()
    for row in data:
        y = wt_phi @ row

        y_hat = wt_phi.T @ y
        
        r.append((y_hat + np.mean(y_hat)) / np.std(y_hat))

    return np.reshape(np.array(r), (-1,size, 1))


y_cA = sensing(train_cA)
y_cD1 = sensing(train_cD1)
y_cD2 = sensing(train_cD2)
y_cD3 = sensing(train_cD3)

testy_cA = sensing(test_cA)
testy_cD1 = sensing(test_cD1)
testy_cD2 = sensing(test_cD2)
testy_cD3 = sensing(test_cD3)

#%% Training

model_cA = CSNet(y_cA, level=2)
model_cD1 = CSNet(y_cD1, level=2)
model_cD2 = CSNet(y_cD2, level=1)
model_cD3 = CSNet(y_cD3, level=0)

epochs = 150

model_cA.train(train_cA, epochs)
model_cA.save('saved_models/alpha_cA.h5')

model_cD1.train(train_cD1, epochs)
model_cD1.save('saved_models/alpha_cD1.h5')

model_cD2.train(train_cD2, epochs)
model_cD2.save('saved_models/alpha_cD2.h5')

model_cD3.train(train_cD3, epochs)
model_cD3.save('saved_models/alpha_cD3.h5')

#%% Inference

predictions_cA = model_cA.predict(test_cA)
predictions_cD1 = model_cD1.predict(test_cD1)
predictions_cD2 = model_cD2.predict(test_cD2)
predictions_cD3 = model_cD3.predict(test_cD3)

#%% Reconstruction

reconstructions = list()

for i in range(predictions_cA.shape[0]):
    coeffs = [predictions_cA[i,:,0], predictions_cD1[i,:,0], predictions_cD2[i,:,0], predictions_cD3[i,:,0]]
    rec = pywt.waverec(coeffs, 'db5')
    
    reconstructions.append(rec)
    
reconstructions = np.array(reconstructions)

#%% Plotting

plots = 15
seed = 504

# for i in range(plots):
    # plt.figure(i+1)
    # plt.plot(test_x[seed+i,:], label='Original data')
    # plt.plot(reconstructions[seed+i,:], label='Reconstructed using CSNet')
    # plt.legend()
    # plt.show()

for i in range(plots):
    plt.figure(i+1)
    plt.plot(y_cD3[seed+i,:,0], label='Original data')
    plt.legend()
    plt.show()

