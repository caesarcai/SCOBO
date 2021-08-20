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
num_iterations = 2000

# Set sampling radius
L = 1
r = np.sqrt(2/L)/2 





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


# Plot the results
init_regret = obj_fcn(x0)
plot_iter = 1540

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
plt.ylim([10**(-1.55),10**(5.5)])
plt.legend(loc="upper right")
plt.grid(True)  
plt.show()