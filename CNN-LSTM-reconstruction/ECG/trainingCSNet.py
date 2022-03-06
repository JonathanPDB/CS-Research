# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 19:39:37 2021

@author: brown
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Activation, LSTM, Dense, Reshape, Permute
from tensorflow.keras.optimizers import Adam
import numpy as np
# import pandas as pd
import pdb
import wfdb
import os 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


#%% Import MIT database

mit_dir = 'C:/Users/brown/OneDrive/IC/MIT_arrhythmia_db/mit-bih-arrhythmia-database-1.0.0'

test_files = [100, 101, 102, 107, 109, \
              111, 115, 117, 118, 119]

test_records = list()
for file in test_files:
    test_record = wfdb.rdsamp(os.path.join(mit_dir, str(file)))
    test_record = test_record[0]
    test_record = np.array(test_record)[:,0]

    test_records.append(test_record)

test_records = np.ravel(np.array(test_records))


train_files = [103,104,105,106,108,112, \
               113,114,116,121,122,123, \
               124,200,201,202,203,205, \
               207,208,209,210,212,213, \
               214,215,217,219,220,221, \
               222,223,228,230,231,232, \
               233,234]                                                                                                                                                                                                                                 

train_records = list()
for file in train_files:
    train_record = wfdb.rdsamp(os.path.join(mit_dir, str(file)))
    train_record = train_record[0]
    train_record = np.array(train_record)[:,0]

    train_records.append(train_record)

train_records = np.ravel(np.array(train_records))

#%% Formatting data


train_size = train_records.shape[0]
N = 256
train_entries = np.floor(train_size / N).astype('int')

excess = train_size % (N * train_entries)

train_x = np.reshape(train_records[excess:], (train_entries, N))   



test_size = test_records.shape[0]
N = 256
test_entries = np.floor(test_size / N).astype('int')

excess = test_size % (N * test_entries)

test_x = np.reshape(test_records[excess:], (test_entries, N))   
 
#%% Compressed sensing

with open('phi/256x32.txt', 'r') as arq:
    phi = arq.readlines()
    
for i, line in enumerate(phi):
    phi[i] = int(line)
    
phi = np.reshape(np.array(phi), (-1,N))
# pdb.set_trace()
r = list()

for training_set in train_x:
    y = phi @ training_set

    y_hat = phi.T @ y
    
    r.append((y_hat + np.mean(y_hat)) / np.std(y_hat))
    
r = np.array(r)
    
#%% CNN

model = Sequential()

r = np.reshape(r, (-1,256,1))
train_x = np.reshape(train_x, (-1,256))

model.add(Conv1D(64, (11), input_shape=r.shape[1:], padding='causal')) 
model.add(Activation('relu'))

model.add(Conv1D(32, (11), padding= 'causal')) 
model.add(Activation('relu'))

model.add(Conv1D(1, (11), padding= 'causal')) 

model.add(Permute((2,1)))

model.add(LSTM(250))
model.add(Activation('tanh'))

model.add(Dense(256))
model.add(Activation('linear'))
model.add(Reshape((256,1)))

model.summary()
# pdb.set_trace()

opt = Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='mean_squared_error',  metrics=['mean_absolute_percentage_error'])

model.fit(r, train_x, batch_size=(256),epochs=300, validation_split=0.125)

model.save('beta_model32_2.h5')

#%% Validation

test_r = list()

for test_set in test_x:
    y = phi @ test_set

    y_hat = phi.T @ y
    
    test_r.append((y_hat + np.mean(y_hat)) / np.std(y_hat))
    
test_r = np.array(test_r)

test_r = np.reshape(test_r, (-1,256,1,1))
test_x = np.reshape(test_x, (-1,256,1))

model.evaluate(test_r, test_x)

res = model.predict(test_r)[:,:,0]

test_x = test_x[:,:,0]

#%% Plotting


seed = 11555

plt.plot(test_x[seed,:], label='Original data')
plt.plot(res[seed,:], label='Reconstructed using CSNet')
plt.legend()
plt.show()





