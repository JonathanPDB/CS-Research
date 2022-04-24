# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 20:10:28 2021

@author: brown
"""

import numpy as np

phi = np.random.choice(2, (256, 16))*2 - 1

with open('random_matrices/phi16.txt', 'w') as arq:
    for j in phi:
        for i in j:
            arq.write(str(i)+'\n')