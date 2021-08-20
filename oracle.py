import numpy as np

class ComparisonOracle(object):
	def __init__(self,obj_fcn,kappa,mu,delta_0):
		self.obj_fcn = obj_fcn
		self.kappa = kappa
		self.mu = mu
		self.delta_0 = delta_0

	def __call__(self,x,y):
	    '''
	    Implements comparison oracle for sparse quadratic
	    In noiseless case, return 1 if f(x)<f(y); otherwise return -1
	    function f(x) = x^TQx
	    May 25th 2020
	    '''
	    fx = self.obj_fcn(x)
	    fy = self.obj_fcn(y)
	    f_diff = np.squeeze(fy - fx)
	    if f_diff == 0:
	        f_diff = (np.random.randint(2) - 0.5)/50
	    prob = 0.5 + np.minimum(self.mu*np.absolute(f_diff)**(self.kappa-1.0),self.delta_0) # Probability of bit-flip
	    mask = 2*np.random.binomial(1,prob) - 1
	    res = np.squeeze(mask*np.sign(f_diff))
	    if mask == 1:
	        bit_flipped = 0
	    else:
	        bit_flipped = 1
	    return res, bit_flipped
