from asyncio.windows_events import NULL
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
import cvxpy as cvx
import scipy.linalg as lng

class ReconAlgs:
    def omp(self, bs):
        skomp = OrthogonalMatchingPursuit(tol = bs.tol, normalize=(False))
        skomp.fit(bs.theta, bs.y)
        i = skomp.n_iter_
        s = skomp.coef_
        return s, i
        
    def cosamp(self, bs, K, maxIter = 1000):
        s = np.zeros(bs.N)
        resid = bs.y
        i = 0  # count
        halt = False

        while not halt:
            i += 1
            corr = np.transpose(bs.theta) @ resid
            
            large_comp = np.argsort(abs(corr))[-(2 * K):]  # large components. Must have abs here so that it finds negative numbers too
            large_comp = np.union1d(large_comp, s.nonzero()[0])  # use set instead?
            ThetaT = bs.theta[:, large_comp]
            
            s = np.zeros(bs.N)
            
            # Solve Least Square
            s[large_comp], _, _, _ = np.linalg.lstsq(ThetaT, bs.y,rcond=-1)
            # s[large_comp] = lng.inv(np.transpose(ThetaT) @ ThetaT) @ np.transpose(ThetaT) @ bs.y
            
            # Get new estimate
            s[np.argsort(abs(s))[:-K]] = 0         # Must also have abs here so that it finds negative numbers too
    
            # Halt criterion
            r_old = resid
            resid = bs.y - bs.theta @ s
    
            halt = ((np.linalg.norm(resid - r_old)) < bs.tol) or \
                   (np.linalg.norm(resid)) < bs.tol or \
                   i > maxIter
                   
        return s, i

    def stomp(self, bs, tStomp, maxIter = 1000,):
        s = np.zeros(bs.N)
        resid = bs.y
        at = []
        i = 0
        halt = False
        
        while not halt:
            i += 1
            
            corr = np.transpose(bs.theta) @ resid
            
            sigma = np.linalg.norm(resid, 2)
            special_thresh = sigma * tStomp
            
            most_corr = np.argwhere(abs(corr) > special_thresh)
            at =  np.union1d(at, most_corr).astype(int)
            ThetaT = bs.theta[:, at]
            
            # s[at], _, _, _ = np.linalg.lstsq(ThetaT, bs.y, rcond=-1)
            s[at] = lng.inv(np.transpose(ThetaT) @ ThetaT) @ np.transpose(ThetaT) @ bs.y
            
            r_old = resid
            resid = bs.y - bs.theta @ s
            
            halt = ((np.linalg.norm(resid - r_old)) < bs.tol) or \
                (np.linalg.norm(resid)) < bs.tol or \
                i > maxIter
        return s, i

    def bp(self, bs):
        s0 = cvx.Variable(bs.N)
        
        objective = cvx.Minimize(cvx.norm1(s0))
        
        constraints = [0.5*(cvx.norm(bs.y-bs.theta @ s0)**2) <= bs.tol] 
        
        prob = cvx.Problem(objective, constraints)
        prob.solve(verbose=False)
        
        res = np.array(s0.value)
        s = np.squeeze(res)
        return s, NULL