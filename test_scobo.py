import numpy as np
import matplotlib.pyplot as plt
from objectives import *
from scobo import *
from oracle import ComparisonOracle
"""
The user MUST install Gurobi to use this program.
Check https://www.gurobi.com/ for installation details.
"""
#'''
################################
# Test problem (a)
# Oracle parameters
kappa = 1.5
mu = 1
delta_0 = 0.5
if kappa == 1:
	fixed_flip_rate = True
else:
	fixed_flip_rate = False

# Dimension and sampling parameters
d = 500
s = 20
m = int(20*s*np.log(2*d/s))

# Setup objective function and the corresponding comparison oracle
obj_fcn = SkewedQuartic(d,s)  # only this pass into SCOBO for recording regret


comparison = ComparisonOracle(obj_fcn,kappa,mu,delta_0)

# Gradient descent parameters
default_step_size = 3
x0 = 50*np.random.rand(d)
num_iterations = 200

# Set sampling radius
r = 1/(2*np.sqrt(s))  
################################
#'''

'''
################################
# Test problem (b)
kappa = 1.5
mu = 4
delta_0 = 0.5
if kappa == 1:
	fixed_flip_rate = True
else:
	fixed_flip_rate = False

d = 500
s = 20
m = int(20*s*np.log(2*d/s))

obj_fcn = MaxK(d,s)
comparison = ComparisonOracle(obj_fcn,kappa,mu,delta_0)


default_step_size = 2

x0 = 20*np.random.randn(d)
num_iterations = 1200

r = 1/(2*np.sqrt(s))/4  
#################################
'''

'''
################################
# Test problem (c)
kappa = 1
mu = 1
delta_0 = 0.3
if kappa == 1:
	fixed_flip_rate = True
else:
	fixed_flip_rate = False


d = 500
s = 20
m = int(20*s*np.log(2*d/s))


obj_fcn = SkewedQuartic(d,s)  # only this pass into SCOBO for recording regret


comparison = ComparisonOracle(obj_fcn,kappa,mu,delta_0)


default_step_size = 2
x0 = 50*np.random.rand(d)
num_iterations = 300

r = 1e-4
################################
'''

'''
################################
# Test problem (d)
kappa = 1
mu = 1
delta_0 = 0.3
if kappa == 1:
	fixed_flip_rate = True
else:
	fixed_flip_rate = False

d = 500
s = 20
m = int(20*s*np.log(2*d/s))

obj_fcn = MaxK(d,s)
comparison = ComparisonOracle(obj_fcn,kappa,mu,delta_0)


default_step_size = 2

x0 = 10*np.random.randn(d)
num_iterations = 800

r = 1e-4
#################################
'''

# Run SCOBO with 3 different settings
line_search = True
warm_started = False
x_hat_linesearch, regret_linesearch,tau_vec_linesearch,c_num_queries_linesearch = SCOBO(comparison,obj_fcn,num_iterations,default_step_size,x0,r,m,d,s,fixed_flip_rate,line_search,warm_started)

line_search = True
warm_started = True
x_hat_warm_started, regret_warm_started,tau_vec_warm_started,c_num_queries_warm_started = SCOBO(comparison,obj_fcn,num_iterations,default_step_size,x0,r,m,d,s,fixed_flip_rate,line_search,warm_started)

line_search = False
warm_started = False
x_hat, regret,tau_vec,c_num_queries = SCOBO(comparison,obj_fcn,num_iterations,default_step_size,x0,r,m,d,s,fixed_flip_rate,line_search,warm_started)



# Plot the SCOBO results with line searched step size
init_regret = obj_fcn(x0)
plot_iter = num_iterations

plt.figure(figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')

BIGGER_SIZE = 20
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE+2)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE-2)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.plot( np.append([0],c_num_queries/1000)[0:num_iterations], np.append([init_regret],regret)[0:num_iterations],'b-', label='Fixed Step Size', linewidth=1)
plt.plot( np.append([0],c_num_queries_linesearch/1000)[0:plot_iter], np.append([init_regret],regret_linesearch)[0:plot_iter],'g-', label='Line Search', linewidth=1)
plt.plot( np.append([0],c_num_queries_warm_started/1000)[0:plot_iter], np.append([init_regret],regret_warm_started)[0:plot_iter],'m-', label='Warm Started Line Search', linewidth=1)
plt.axhline(y=1.5*default_step_size, color='y', linestyle='-.',label='Theoretical error bound')
plt.xlabel(r'Number of comparison oracles $\times 10^3$')
plt.ylabel('Optimality gap')
plt.yscale('log')
#plt.ylim([10**(-1.55),10**(5.5)])
plt.legend(loc="upper right")
plt.grid(True)  
plt.show()



# Plot the SCOBO results with fixed step size
fig, ax1 = plt.subplots(figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')

color = 'tab:blue'
ax1.set_xlabel(r'Number of comparison oracles $\times 10^3$')
ax1.set_ylabel('Optimality gap', color=color)
ax1.set_yscale('log')
#ax1.set_ylim([10**(-1.55),10**(5.5)])
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