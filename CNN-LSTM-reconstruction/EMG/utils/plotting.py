import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

def seedPlot(original, recon, seed):

    if type(seed) == int:
        seed = np.random.randint(1, original.shape[0], size=seed)

    f = 1
    for s in seed:
        plt.figure(f)
        f += 1
        
        plt.plot(original[s,:], label='Original data')
        plt.plot(recon[s,:], label='Reconstructed using CSNet')
        plt.legend()
        
        
    plt.show()
