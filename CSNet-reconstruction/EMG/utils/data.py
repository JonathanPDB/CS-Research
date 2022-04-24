import numpy as np
import scipy.io


class Data:  
    DATABASE_PATH = 'database/sEMG/S1_A1.mat'

    def importData(self, set):
        S1_A1 = scipy.io.loadmat(self.DATABASE_PATH)
        sEMG = S1_A1['emg']
        
        if set == 'test':
            files = np.transpose(sEMG[:,0:2])
        
        elif set == 'train':
            files = np.transpose(sEMG[:,2:8])

        return files

    def splitData(self, records, N):   
        x = list()
        for row in records:
            size = row.shape[0]
            entries = np.floor(size / N).astype('int')
            excess = size % (N * entries)

            x.append(np.reshape(row[excess:], (entries, N)))

        x = np.array(x)
        return  np.reshape(x, (x.shape[0]*x.shape[1], x.shape[2])) 
 


