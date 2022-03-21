import os
import numpy as np
from tensorflow.keras import models as m
from tensorflow.keras.layers import Conv1D, Activation, Permute, LSTM, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from datetime import datetime

MODEL_BASEPATH = 'saved_models'

class CSNet():
    def __init__(self, N):
        self.N = N

    def createModel(self, x):
        model = m.Sequential()

        self.x = np.reshape(x, (-1,self.N,1))

        model.add(Conv1D(64, (11), input_shape=self.x.shape[1:], padding='causal')) 
        model.add(Activation('relu'))

        model.add(Conv1D(32, (11), padding= 'causal')) 
        model.add(Activation('relu'))

        model.add(Conv1D(1, (11), padding= 'causal')) 

        model.add(Permute((2,1)))

        model.add(LSTM(250))
        model.add(Activation('tanh'))

        model.add(Dense(256))
        model.add(Reshape((256,1)))

        model.summary()

        self.model = model

    def train(self, gold, epochs=20):
        
        self.gold = np.reshape(gold, (-1,256))

        opt = Adam(learning_rate=0.0005)
        self.model.compile(optimizer=opt, loss='mean_squared_error',  metrics=['mean_absolute_percentage_error'])
        self.model.fit(self.x, self.gold, batch_size=(self.N),epochs=epochs, validation_split=0.125)

        self.model.save(self.getSaveName())

    def getSaveName(self):
        utc = datetime.now().strftime("%m-%d-%Y_%H-%M")
        self.saveName = 'EMG_' + utc + '.h5'

        return os.path.join(MODEL_BASEPATH, self.saveName)


def load(modelPath):
    path = os.path.join(MODEL_BASEPATH, modelPath)
    return m.load_model(path)
    
def evaluate(model, inputSet, goldenSet):
    model.predict(inputSet, goldenSet)
    
def predict(model, inputSet):
    return model.predict(inputSet)[:,:,0]