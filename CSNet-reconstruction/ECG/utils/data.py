import wfdb
import os
import numpy as np


class Data:  
    DATABASE_PATH = '../../../../CS-Research-databases/ECG/mit-bih-arrhythmia-database-1.0.0'

    TEST_SET = [100, 101, 102, 107, 109, \
                111, 115, 117, 118, 119]

    TRAINING_SET = [103,104,105,106,108,112, \
                    113,114,116,121,122,123, \
                    124,200,201,202,203,205, \
                    207,208,209,210,212,213, \
                    214,215,217,219,220,221, \
                    222,223,228,230,231,232, \
                    233,234]

    def importData(self, set):
        if set == 'test':
            files = self.TEST_SET
        
        elif set == 'train':
            files = self.TRAINING_SET

        selectedRecords = list()

        for file in files:
            record = wfdb.rdsamp(os.path.join(self.DATABASE_PATH, str(file)))
            record = record[0]
            record = np.array(record)[:,0]

            selectedRecords.append(record)

        return np.ravel(np.array(selectedRecords))

    def splitData(self, raw, N):   
        size = raw.shape[0]
        entries = np.floor(size / N).astype('int')
        excess = size % (N * entries)
        return np.reshape(raw[excess:], (entries, N))   
 


