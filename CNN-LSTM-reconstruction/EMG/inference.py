#%% Imports 
from utils import CS, Data
import utils.plotting as p
import CSNet.model as m


#%% Data

d = Data()
records = d.importData('test')

N = 256
x = d.splitData(records, N) 

#%% Compressed sensing

cs = CS()
phiName = 'phi32.txt'

phi = cs.randomMatrix(phiName, N)
r = cs.compress(x)


#%% CSNet

modelName = 'beta_EMG32_256.h5'
infered = m.load(modelName).predict(r)

#%% Plotting

p.seedPlot(x, infered, 10)
