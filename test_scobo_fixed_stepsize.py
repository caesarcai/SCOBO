import numpy as np
import matplotlib.pyplot as plt
from testfunctions import *
from scobo import *
from oracle import ComparisonOracle
#The user must install Gurobi to use this program


# Oracle parameters
kappa = 2
mu = 1
delta_0 = 0.5
if kappa == 1:
	fixed_flip_rate = True
else:
	fixed_flip_rate = False

# Dimension and sampling parameters
d = 500
m = 200
s = 20

# Setup objective function and the corresponding comparison oracle
obj_fcn = SkewedQuartic(d,s)  # only this pass into SCOBO for recording regret
comparison = ComparisonOracle(obj_fcn,kappa,mu,delta_0)

# Gradient descent parameters
default_step_size = 1  #10
x0 = 100*np.ones((d,1))
init_regret = obj_fcn(x0)
num_iterations = 2000
plot_iter = num_iterations

# Set sampling radius
L = 1
r = np.sqrt(2/L)/2 

'''
################################
# Another test problem
kappa = 1.5
mu = 4
delta_0 = 0.5
if kappa == 1:
	fixed_flip_rate = True
else:
	fixed_flip_rate = False

d = 500
s = 20
m = int((s**2)*np.log(2*d/s))

obj_fcn = MaxK(d,s)
comparison = ComparisonOracle(obj_fcn,kappa,mu,delta_0)


default_step_size = 2

x0 = 20*np.random.randn(d)
x0_backup=np.copy(x0)
num_iterations = 1200

L = 1
r = 1/(2*np.sqrt(s))/4  
#################################
'''

line_search = False
warm_started = False
x_hat, regret,tau_vec,c_num_queries = SCOBO(comparison,obj_fcn,num_iterations,default_step_size,x0,r,m,d,s,fixed_flip_rate,line_search,warm_started)


# Plot the results
fig, ax1 = plt.subplots(figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')

color = 'tab:blue'
ax1.set_xlabel(r'Number of comparison oracles $\times 10^3$')
ax1.set_ylabel('Optimality gap', color=color)
ax1.set_yscale('log')
ax1.set_ylim([10**(-1.55),10**(5.5)])
plt.plot( np.append([0],c_num_queries/1000)[0:plot_iter], np.append([init_regret],regret)[0:plot_iter],'b-', label='Fixed Step Size', linewidth=1)
plt.axhline(y=1.5*default_step_size, color='y', linestyle='-.',label='Theoretical error bound')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Fraction flipped measurement', color=color)  # we already handled the x-label with ax1
plt.plot( np.append([0],c_num_queries/1000)[0:plot_iter],np.append([0],tau_vec)[0:plot_iter],'r-', label='Fixed Step Size', linewidth=0.4)
ax2.set_ylim([-0.025,0.575])
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.legend(loc="upper right")
plt.grid(True)
plt.show()