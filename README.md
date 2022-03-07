# CS-Research

This repository is used for researching and implementing state of the art approaches on Compressed Sensing.

As there are inumerous methods and applications, the different files represent a different type of implementation. 
The objective of this research is to further explore the capabilities of CS, as well as it's limitations.

As of now, there are two main focus point in this repository.

### Traditional Compressed Sensing

The objective is to implement the standard and most well known Compressed Sensing methods as Proof of Concept of the base idea behind it.

Therefore, standard signals, basis and reconstruction algorithms were implemented.

### CNN and LSTM approach for biosignal sampling

This method is more specific and aimed at sampling and reconstructing biosignals, i.e., ECG and EMG.

The ECG repo is focused on replicating the article written by Zhang et al. (2019) [CSNet: A deep learning approach for ECG compressed sensing](https://www.sciencedirect.com/science/article/pii/S1746809421006625 "CSNet").

On the other hand, the EMG repo attempts to further the latter article and apply the technique on sEMG signals.