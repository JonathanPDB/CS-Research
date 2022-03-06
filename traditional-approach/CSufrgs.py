# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 12:05:52 2021

@author: brown
"""

import numpy as np
import scipy.linalg as lng
from scipy.fftpack import rfft, irfft, dct, idct
from scipy import stats
from scipy.signal import sawtooth, chirp
import cvxpy as cvx
import pdb
from datetime import datetime
from scipy.misc import electrocardiogram
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.optimize import least_squares
from FuzMP import fuzmp as FuzMP

class signal():
    def add_noise(x, SNR):
        N = x.shape[0]
    
        Px_dB = 10 * np.log10(np.mean(x**2))
            
        noise_dB = Px_dB - SNR
        noise_t = 10 ** (noise_dB / 10)
        
        noise = np.random.normal(0, np.sqrt(noise_t), N)
        x += noise
        
        return x
    
    def spikes(N, K, SNR=0, f_max=None, f_min=1, a_max=20, a_min=10):
        x = np.zeros(N)

        if f_max == None:
            f_max = np.floor(N/2 - 1)
        # pdb.set_trace()
        peaks = np.random.randint(a_min,a_max, size=K)
        freqs = np.random.randint(f_min,f_max, size=K)
        
        for i in range(freqs.shape[0]):
            if i % 2:
                x[freqs[i]] = peaks[i]
            else:
                x[freqs[i]] = -peaks[i]
    
        xclean = x
        
        if SNR:
            x = signal.add_noise(x, SNR)   
    
        return x, xclean
    
    def sinusoidal(N, K, SNR=0, f_max=None, f_min=1, a_max=20, a_min=10):
        f = np.zeros(N)

        if f_max == None:
            f_max = np.floor(N/2 - 1)
        # pdb.set_trace()
        peaks = np.random.randint(a_min,a_max, size=K)
        freqs = np.random.randint(f_min,f_max, size=K)
        
        for i in range(freqs.shape[0]):
            if i % 2:
                f[freqs[i]] = peaks[i]
            else:
                f[freqs[i]] = -peaks[i]
        
        x = irfft(f, N)
        x /= N/2
        
        xclean = x
        
        if SNR:
            x = signal.add_noise(x, SNR)   
    
        return x, xclean
    
    def saw(N, w=0.5, a_max = 1, SNR=0, K=1):
        
        wind = int(np.floor(N/K))
        
        t = np.linspace(0, 1, N)
        tK = np.linspace(0, 1, wind)
        
        sawK = sawtooth(2*np.pi*tK, w) * a_max
        
        # pdb.set_trace()
        
        x = np.zeros(N)
        
        for i in range(K):
            x[i*wind:wind*(i+1)] += sawK
           
        diff = N - wind*K
        if diff > 0:
            x = x[:wind*K]
            x = np.pad(x, (0, diff), 'constant', constant_values=sawK[-1])
            
        xclean = x
        
        if SNR:
             x = signal.add_noise(x, SNR)  
         
        return x, xclean
    
    def normal(N, a_max = 1, mean=0.5, std=0.01, SNR=0, K=1):
        mean = 1/(K+1)
        meanK = mean
        
        t = np.linspace(0,1, N)
        
        x = np.zeros(N)
        
        for i in range(K):
            xK = stats.norm.pdf(t,meanK,std)*a_max
            x += xK
            
            meanK += mean
           
        
        xclean = x
        
        if SNR:
             x = signal.add_noise(x, SNR)  
         
        return x, xclean
    
    def chirper(N, SNR=0, f_min =1, f_max = 10):
        t = np.linspace(0,1 ,N)
        x = chirp(t, f0=f_min, f1=f_max, t1=1, method='linear')
        
        xclean = x
        
        if SNR:
             x = signal.add_noise(x, SNR)  
         
        return x, xclean

    def ecg(N, SNR=0, start_point=4000):
        x = electrocardiogram()[start_point:(start_point+N)]
        xclean = x
        
        if SNR:
            x = signal.add_noise(x, SNR)
            
        return x, xclean
    
    def emg(N, SNR=0):
        
        if N > 4096:
            raise ValueError('N must not exceed 4096 for EMG signals')
            return
        
        with open('semg.txt', 'r') as arq:
            emg_strings = arq.readlines()
        
        x = np.zeros(N)
        
        for index, string in enumerate(emg_strings[0:N]):
            x[index] = float(string)
            
        xclean = x
        
        if SNR:
            x = signal.add_noise(x, SNR)
            
        return x, xclean


class compress():
    def __init__(self, x, M, measurement='bernoulli', basis='fft'):
        
        self.x = x
        self.N = x.shape[0]
        self.M = M
        self.basis = basis
        
        self.start = datetime.now()
        
        if measurement == 'gaussian':
            self.gaussian()
            
        elif measurement == 'toeplitz':
            self.toeplitz()
            
        elif measurement == 'bernoulli':
            self.bernoulli()
            
        elif measurement == 'uniform':
            self.uniform()
            
        elif measurement == 'fixed':
           self.fixed()
           
        else:
            raise ValueError('Measurement parameter not valid')
        
        self.y = self.phi @ self.x
        
        
        if basis == 'fft':
            self.fft()
        elif basis == 'sparse':
            self.sparse()
        else:
            self.psi = np.transpose(basis)
        # else:
        #     raise ValueError('Transform basis parameter not valid')
        
        self.theta = self.phi @ self.psi        
        
    
            
    def normalize(self):
        rows    = np.split(self.phi, self.M)
        
        matrix = []
        for row in rows:
            norm = np.linalg.norm(row)
            if norm==0:
                matrix.append(row)
            else:
                matrix.append(np.squeeze(row/norm))
        
        self.phi = np.reshape(matrix, (self.M, self.N))
    
    def gaussian(self):
        self.phi = np.random.randn(self.M, self.N)
        self.normalize()
    
    def toeplitz(self):
        sample_indices = np.random.choice(2, self.N)*2 - 1
        toe = lng.toeplitz(sample_indices)
        random_indices = np.random.choice(self.N, self.M, replace=False)
        
        self.phi = toe[random_indices,:]
        self.normalize()
    
    def bernoulli(self):
        self.phi = np.random.choice(2, (self.M, self.N))*2 - 1
        self.normalize()
    
    def uniform(self):
        random_indices = np.random.choice(self.N, self.M, replace=False)
        self.phi = np.zeros((self.M,self.N))
        self.phi[range(self.M),random_indices] = 1
        
    def fixed(self):
        if self.M > 1024:
            raise ValueError('M must not exceed 1024 for fixed phi')
            return
        
        with open('fixed_random.txt', 'r') as arq:
            fixed_strings = arq.readlines()
        
        random_indices = np.zeros(self.M)
        
        for index, string in enumerate(fixed_strings[0:self.M]):
            random_indices[index] = int(string)
        
        random_indices = random_indices.astype(int)
        
        self.phi = np.zeros((self.M,self.N))
        self.phi[range(self.M),random_indices] = 1

    def fft(self):
        self.psi = rfft(np.identity(self.N))
        
    def sparse(self):
        self.psi = np.identity(self.N)
    
    def dcost(self):
        self.psi = dct(np.identity(self.N))


class recon():
    def __init__(self, compression, alg, K=10, tol=1e-10, max_iter=1000,\
                 t_stomp = 2.0, threshold=0, offset=0, verbose=False, fuzzy=0):
        
        if not isinstance(compression, compress):
            raise TypeError('Compression parameter has to be an object of the compress class.')
        
        self.Theta = compression.theta
        self.y = compression.y
        self.N = compression.N
        self.M = compression.M
        self.x = compression.x
        self.i = None
        
        start = datetime.now()
        
        if alg == 'cosamp' or alg == 0:
            self.cosamp(K, max_iter, tol)
        
        elif alg == 'omp' or alg == 1:
            self.omp(K, max_iter, tol)
            
        elif alg == 'stomp' or alg == 2:
            self.stomp(t_stomp, max_iter, tol)
        
        elif alg == 'bp' or alg == 3:
            self.bp(tol)
            
        elif alg == 'omp2' or alg == 4:
            self.omp2(K, tol)
            
        elif alg == 'omp3' or alg == 5:
            self.omp3(K, max_iter, tol)
        
        elif alg == 'fuzmp' or alg == 6:
            self.fuzmp(K, max_iter, tol, fuzzy)
        
        else:
            raise ValueError('TReconstruction algorithm parameter not valid')
         
        
        if threshold:
            max_s = np.max(abs(self.s))
            ts = threshold*max_s
            self.s[np.where(abs(self.s) < ts)] = 0    
         
        self.reconstruct(compression.basis)
        
        if offset:
            self.xrec += offset
        
        self.end = datetime.now()
        self.recontime = (self.end - start).total_seconds()
        
        if verbose:
            print('Recontruction time:', self.recontime)
            print('Iterations:', self.i)
        
    def cosamp(self, K, max_iter, tol):

        s = np.zeros(self.N)
        resid = self.y
        i = 0  # count
        halt = False
        while not halt:
            i += 1
            # pdb.set_trace()
            corr = np.transpose(self.Theta) @ resid
            
            mean = np.mean(abs(corr))
            std = np.std(abs(corr))
            topmean = np.mean(np.sort(abs(corr))[-100:])
            max_val = abs(corr).max()
            norm = np.random.normal(mean, std, 1000)
            corrsorted = np.sort(abs(corr))
            
            
            large_comp = np.argsort(abs(corr))[-(2 * K):]  # large components. Must have abs here so that it finds negative numbers too
            large_comp = np.union1d(large_comp, s.nonzero()[0])  # use set instead?
            ThetaT = self.Theta[:, large_comp]
            
            s = np.zeros(self.N)
            
            # Solve Least Square
            # s[large_comp], _, _, _ = np.linalg.lstsq(ThetaT, self.y,rcond=-1)
            s[large_comp] = lng.inv(np.transpose(ThetaT) @ ThetaT) @ np.transpose(ThetaT) @ self.y
            
            # Get new estimate
            s[np.argsort(abs(s))[:-K]] = 0         # Must also have abs here so that it finds negative numbers too
    
            # Halt criterion
            r_old = resid
            resid = self.y - self.Theta @ s
    
            halt = ((np.linalg.norm(resid - r_old)) < tol) or \
                   (np.linalg.norm(resid)) < tol or \
                   i > max_iter
        
        self.s = s
        self.i = i
        
    def omp(self, K, max_iter, tol):
        
        resid = self.y
        at = []
        i = 0
        halt = False
        # print('a')
        while not halt:
            i += 1
            # pdb.set_trace()
            
            corr = np.transpose(self.Theta) @ resid
            
            most_corr = abs(corr).argmax()
            at = np.append(at, most_corr).astype(int)
            ThetaT = self.Theta[:, at]
            
            s = np.zeros(self.N)
            s2 = np.zeros(self.N)
            
            s[at], _, _, _ = np.linalg.lstsq(ThetaT, self.y, rcond=-1)
            # s[at] = lng.inv(np.transpose(ThetaT) @ ThetaT) @ np.transpose(ThetaT) @ self.y
            
            r_old = resid
            resid = self.y - self.Theta @ s
            
            halt = ((np.linalg.norm(resid - r_old)) < tol) or \
                (np.linalg.norm(resid)) < tol or \
                i > max_iter
        
        self.i = i
        self.s = s
        
        
        
    def fuzmp(self, K, max_iter, tol, t_stomp):
        
        self.s, self.i = FuzMP(self.y, self.Theta, K, max_iter, tol, t_stomp)
        
        
        
    def stomp(self, t_stomp, max_iter, tol):
        
        s = np.zeros(self.N)
        resid = self.y
        at = []
        i = 0
        halt = False
        
        while not halt:
            i += 1
            
            # pdb.set_trace()
            
            corr = np.transpose(self.Theta) @ resid
            
            sigma = np.linalg.norm(resid, 2)
            special_thresh = sigma * t_stomp
            
            most_corr = np.argwhere(abs(corr) > special_thresh)
            at =  np.union1d(at, most_corr).astype(int)
            ThetaT = self.Theta[:, at]
            
            # s[at], _, _, _ = np.linalg.lstsq(ThetaT, self.y, rcond=-1)
            s[at] = lng.inv(np.transpose(ThetaT) @ ThetaT) @ np.transpose(ThetaT) @ self.y
            
            r_old = resid
            resid = self.y - self.Theta @ s
            
            halt = ((np.linalg.norm(resid - r_old)) < tol) or \
                (np.linalg.norm(resid)) < tol or \
                i > max_iter
        
        self.s = s
        self.i = i
    
    def bp(self, tol):

        s0 = cvx.Variable(self.N)
        objective = cvx.Minimize(cvx.norm1(s0))
        
        constraints = [0.5*(cvx.norm(self.y-self.Theta @ s0)**2) <= tol] 
        
        prob = cvx.Problem(objective, constraints)
        prob.solve(verbose=False)
        
        res = np.array(s0.value)
        self.s = np.squeeze(res)

        
    def omp2(self, K, tol):
        
        skomp = OrthogonalMatchingPursuit(n_nonzero_coefs=K, tol = tol)
        skomp.fit(self.Theta, self.y)
        self.i = skomp.n_iter_
        self.s = skomp.coef_
        
        
    def reconstruct(self, basis):
        if basis == 'fft':
            self.xrec = irfft(self.s)
            self.xrec *= self.N/2  
        
        elif basis == 'dct':
            self.xrec = idct(self.s)
            # self.xrec *= 2 
        
        elif basis == 'sparse':
            self.xrec = self.s
        else:
            self.xrec = basis @ self.s
    
    
class metrics():
    def __init__(self, compression, rec, verbose=True, plots=False):
            
        
        self.runtime = (rec.end - compression.start).total_seconds()
        
        self.MSE = (np.linalg.norm(rec.x-rec.xrec)**2)/np.linalg.norm(rec.x)**2
        
        self.MSE = 10 * np.log10(self.MSE)
        
        self.reconerror = np.linalg.norm(rec.x-rec.xrec, 1)/np.linalg.norm(rec.x,1)
        
        if verbose:
            print('Total runtime:', self.runtime)
            print('MSE:', self.MSE)
            print('Reconstruction Error:', self.reconerror)
            if rec.i:
                print('Number of iterations: ', rec.i)
        
        if plots:
            self.plots()
            
            return self