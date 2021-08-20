'''
This module contains the following:

Sparse Quadric

Max-k-sum-squared

SkewedQuartic

'''
import numpy as np


class SparseQuadric(object):
    '''An implementation of the sparse quadric function.'''
    def __init__(self, d, s):
        self.s = s
        self.dim = d
        
    def __call__(self,x):
        return np.dot(x[0:self.s],x[0:self.s])
    

class MaxK(object):
    '''An implementation of the max-k-squared-sum function.'''
    def __init__(self, d, s):
        self.dim = d
        self.s = s

    def __call__(self, x):
        idx = np.argsort(np.abs(x))
        idx2 = idx[self.dim-self.s:self.dim]
        return np.dot(x[idx2], x[idx2])/2
 


class SkewedQuartic(object):
    '''An implementation of the sparse quadric function.'''
    def __init__(self, d, s):
        self.dim = d
        self.s = s
        Diagonal = np.concatenate((np.ones(s),np.zeros(d-s)))
        self.B = np.diag(Diagonal)
        self.BB= np.dot(self.B.T, self.B)
        
    def __call__(self, x):
        return np.dot(np.dot(x.T,self.BB),x) + 0.1*np.sum(np.power(np.dot(self.B,x),3)) + 0.01*np.sum(np.power(np.dot(self.B,x),4))
    