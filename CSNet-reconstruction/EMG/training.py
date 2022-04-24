#%% Imports 

from utils import Data, CS
from CSNet.model import CSNet

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

csnet = CSNet(N)
csnet.createModel(r)
csnet.train(gold=x, epochs=1)

print('\nModel saved as', csnet.saveName)
