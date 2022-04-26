# CS Net

## Objective

CSNet is an approach based on CNN and LSTM proposed by Zhang et al. (2021) [[1]](https://www.sciencedirect.com/science/article/pii/S1746809421006625). It was developed with the objective of reconstructing ECG signals with a very high accuracy and very low sampling size. 

Due to it's impressive results and promising idea, we have attempted to replicate the published work and also expand the concept for sEMG reconstruction. 

The results obtained so far are not nearly as good as the ones shown in the article (we have compared the same ECG entries and with what we believe to be the same conditions and parameters). Therefore, we are still in the process of managing such replication.

---
## Proposed methodology

In order to better visualize all the steps described in [[1]](https://www.sciencedirect.com/science/article/pii/S1746809421006625) the proposed model will be sumarized here. 


### Database

The MIT-BIH arrhythmia was used, which contains 48 30-min-long records with two leads (MLII and V5) for each record. Only MLII data is used in this experiment, and the data sampling rate is 360 Hz. Below are the records that were used for each set.

* Training Set
```
103, 104, 105, 106, 108, 112, 113, 114, 116, 121, 122, 123,
124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213,
214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232,
233, 234
```
* Test Set
```
100, 101, 102, 107, 109, 111, 115, 117, 118, 119
```

In addition to this, 12.5% of the training set is randomly chosen as a validation set, resulting in a 7:1:2 proportion (training, validation, testing).

### Sampling

As will be seen later, the ML model expects a fixed vector size as an input. This means that regardless of the compression ratio that will be used **M** must be the same. The way this is achieved is by performing 2 matrix producs.

![equation](https://latex.codecogs.com/svg.image?\bg{white}y&space;=&space;\phi_{N&space;\times&space;M}&space;\times&space;x)
 
![equation](https://latex.codecogs.com/svg.image?\bg{white}\hat{y}&space;=&space;\phi_{N&space;\times&space;M}^T&space;\times&space;y)

Where ŷ is the randomly sampled vector. 

Note: M is the orignal size of the signal vector and the size that will be fed into the ML model. N is the ammount of samples, e.g. M = 256 and N = 32.

### Preprocessing

The data is normalized using the standard distribution. 

![equation](https://latex.codecogs.com/svg.image?\bg{white}r&space;=&space;\frac{\hat{y}&space;-&space;\hat{y}_\mu}{\hat{y}_\sigma})


### Model

The reconstruction algorithm has 2 stages: CNN and LSTM.

Despite the article not being exactly clear wether these algorithms are trained seperately, we belive there is only one training proccess, meaning that the output of the CNN will be the input of the LSTM. 

| **Layer**   | **Feature Map** | **Output Shape** |
|-------------|-----------------|------------------|
| Convolution | 64              | 256x64           |
| ReLu        | -               | 256x64           |
| Convolution | 32              | 256x32           |
| ReLu        | -               | 256x32           |
| Convolution | 1               | 256x1            |
| Permute     | -               | 1x256            |
| LSTM        | -               | 1x250            |
| Tanh        | -               | 1x250            |
| Dense       | -               | 256              |
| Reshape     | -               | 256x1            |

**Important Parameters**

* The kernel size of the CNN layers is 11x1.
* The Loss function is MSE
* Adam was the chosen Optimizer


The model contains many layers, which is something to look into, as one of the problem we are encontering at the moment is the model not being able to learn properly, especially the LSTM layer. This will be discussed in a further section.

Another thing to notice is that the output signal is basically the reconstructed normalized signal, without the added need of a inverse transform. This many be due to the fact that ECG signals can be sparse in time domain, but not always, especially with a 256 sample time-frame.


###  Metrics

* Compression Ratio

![equation](https://latex.codecogs.com/svg.image?CR&space;=&space;\frac{m-n}{m}.100%)


* Percentage Root-mean-square Difference (PRD)

![equation](https://latex.codecogs.com/svg.image?\bg{white}PRD&space;=&space;\frac{||x-\hat{x}||_2}{||x||_2}.100%)

* Signal-to-Noise Ratio (SNR)

![equation](https://latex.codecogs.com/svg.image?\bg{white}PRD&space;=&space;10log_{10}\frac{||x||_2}{||x-\hat{x}||_2})


## Results

### Original

The results of the original paper are impressive to say the least. 

With a CR of 90% (aproximately 26 samples out of 256) CSNet obtained a PRD below 9 during a 5 second window. This result completely out performed algorithms such as OMP and BP, which didn't manage to reconstruct anything at all and outputed only noise.

**Optimal Compression Ratio for each method**
| **Records** | **BP** | **OMP** | **BSBL-BO** | **R-SVD+BP** | **CSNet** |
|------|------|-----|-----|------|------| 
| 100          | 47% | 72% | 71% | 72% | 88% | 
| 101          | 48% | 71% | 71% | 72% | 88% | 
|102          | 48% | 65% | 70% | 59% | 63% | 
|107          | 56% | 70% | 74% | 78% | 81% | 
|109          | 53% | 72% | 75% | 84% | 87% | 
|111          | 47% | 52% | 69% | 74% | 80% | 
|115          | 49% | 71% | 68% | 81% | 91% | 
|117          | 64% | 73% | 74% | 91% | 93% | 
|118          | 49% | 72% | 76% | 83% | 87% | 
|119          | 57% | 73% | 71% | 88% | 92% | 
|Average      | 52% | 69% | 72% | 78% | 85% | 


## References

[1] [CSNet: A deep learning approach for ECG compressed sensing](https://www.sciencedirect.com/science/article/pii/S1746809421006625 "CSNet")