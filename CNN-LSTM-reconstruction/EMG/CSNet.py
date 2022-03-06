# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:39:21 2021

@author: brown
"""

import pandas as pd
import numpy as np
import tensorflow as tf


class CSNet:
    def __init__(self, y, level=0):
        self.y = y
        self.level = level
        self.N = y.shape[1]
        
        self.create_model()

        
    def create_model(self):
        shape = self.y.shape[1:]
        
        model = tf.keras.models.Sequential()
        
        model.add(tf.keras.layers.Input(shape=shape))
        
        if not self.level:
            model.add(tf.keras.layers.Conv1D(64, (11), padding='causal', activation='relu'))
            
        # if self.level <= 1:    
        model.add(tf.keras.layers.Conv1D(32, (11), padding='causal', activation='relu'))
            
        # model.add(tf.keras.layers.Conv1D(16, (11), padding='causal', activation='relu'))
        
        model.add(tf.keras.layers.Conv1D(1, (11), padding='causal', activation='relu'))
        
        model.add(tf.keras.layers.Permute((2,1)))

        model.add(tf.keras.layers.LSTM(self.N - 4))
        model.add(tf.keras.layers.Activation('tanh'))

        model.add(tf.keras.layers.Dense(self.N))
        model.add(tf.keras.layers.Activation('linear'))
        model.add(tf.keras.layers.Reshape((self.N,1)))
        
        model.summary()

        opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=opt, loss="mse",  metrics=["mean_absolute_percentage_error"])

        self.model = model
        

    def train(self, labels, epochs):
        self.model.fit(self.y, labels, batch_size=(self.N), epochs=epochs, validation_split=0.125)

    def save(self, file_path):
        self.model.save(file_path)
        
    def predict(self, x):
        return self.model.predict(x)