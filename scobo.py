import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp 
from gurobipy import GRB, quicksum
#The user must install Gurobi to use this program


def Solve1BitCS(y,Z,m,d,s):
    '''
    This function creates a quadratic programming model, calls Gurobi
    and solves the 1 bit CS subproblem. This function can be replaced with
    any suitable function that calls a convex optimization package.
    =========== INPUTS ==============
    y ........... length d vector of one-bit measurements
    Z ........... m-by-d sensing matrix
    m ........... number of measurements
    d ........... dimension of problem
    s ........... sparsity level
    
    =========== OUTPUTS =============
    x_hat ....... Solution. Note that ||x_hat||_2 = 1
    '''   
    model = gp.Model("1BitRecovery")
    x = model.addVars(2*d, vtype = GRB.CONTINUOUS)
    c1 = np.dot(y.T,Z)
    c = list(np.concatenate((c1,-c1)))

    model.setObjective(quicksum(c[i]*x[i] for i in range(0,2*d)), GRB.MAXIMIZE)
    model.addConstr(quicksum(x) <= np.sqrt(s),"ell_1")  # sum_i x_i <=1
    model.addConstr(quicksum(x[i]*x[i] for i in range(0,2*d)) - 2*quicksum(x[i]*x[d+i] for i in range(0,d))<= 1, "ell_2") # sum_i x_i^2 <= 1
    model.addConstrs(x[i] >= 0 for i in range(0,2*d))
    model.Params.OUTPUTFLAG = 0

    model.optimize()
    TempSol = model.getAttr('x')
    x_hat = np.array(TempSol[0:d] - np.array(TempSol[d:2*d]))
    return x_hat


def GradientEstimator(Comparison,x_in,Z,r,m,d,s):
    '''
    This function estimates the gradient vector from m Comparison
    oracle queries, using 1 bit compressed sensing and Gurobi
    ================ INPUTS ======================
    Comparison............. comparison orcale
    x_in .................. Any point in R^d
    Z ..................... An m-by-d matrix with rows z_i uniformly sampled from unit sphere
    r ..................... Sampling radius.
    kappa,delta_0, mu...... Comparison oracle parameters.
    m ..................... number of measurements.
    d ..................... dimension of problem
    s ..................... sparsity
    
    ================ OUTPUTS ======================
    g_hat ........ approximation to g/||g||
    tau .......... fraction of bit-flips/ incorrect one-bit measurements.
    '''
    y = np.zeros(m)
    tau = 0
    for i in range(0,m):
        x_temp = Z[i,:]
        y[i], bit_flipped = Comparison(x_in,x_in + r*Z[i,:])
        tau += bit_flipped
    g_hat = Solve1BitCS(y,Z,m,d,s)
    tau = tau/m
    return g_hat,tau


def GetStepSize(Comparison,x,g_hat,last_step_size,default_step_size,warm_started):
    '''
    This function use line search to estimate the best step size on the given 
    direction via noisy comparison 
    ================ INPUTS ======================
    Comparison................ comparison orcale
    x ........................ current point
    g_hat .................... search direction
    last_step_size ........... step size from last itertion
    default_step_size......... a safe lower bound of step size
    kappa,delta_0, mu......... Comparison oracle parameters.
    
    ================ OUTPUTS ======================
    alpha .................... step size found
    less_than_defalut ........ return True if found step size less than default step size
    queries_count ............ number of oracle queries used in linesearch
    '''
    
    # First make sure current step size descends
    omega = 0.05
    num_round = 40
    descend_count = 0
    queries_count = 0
    less_than_defalut = False
    #update_factor = np.sqrt(2)
    update_factor = 2
    
    if warm_started:
        alpha = last_step_size  # start with last step size
    else:
        alpha = default_step_size
    point1 = x - alpha * g_hat
    
    for round in range(0,num_round): # compare n rounds for every pair of points, 
        is_descend,bit_flipped = Comparison(point1,x)
        queries_count = queries_count + 1
        if is_descend == 1:
            descend_count = descend_count + 1
    p = descend_count/num_round
    # print(p)
    
    
    # we try increase step size if p is larger, try decrease step size is
    # smaller, otherwise keep the current alpha
    if p >= 0.5 + omega:   # compare with x
        while True:        
            point2 = x - update_factor * alpha * g_hat
            descend_count = 0
            for round in range(0,num_round):   # compare n rounds for every pair of points,
                is_descend,bit_flipped = Comparison(point2,point1)   # comapre with point1
                queries_count = queries_count + 1
                if is_descend == 1:
                    descend_count = descend_count + 1
            p = descend_count/num_round
            if p >= 0.5 + omega:
                alpha = update_factor * alpha
                point1 = x - alpha * g_hat
            else:
                return alpha,less_than_defalut,queries_count
    elif warm_started == False:
        less_than_defalut = True
        return alpha,less_than_defalut,queries_count
    elif p <= 0.5 - omega:   # else: we try decrease step size
        while True:
            alpha = alpha / update_factor
            if alpha < default_step_size:
                alpha = default_step_size
                less_than_defalut = True
                return alpha,less_than_defalut,queries_count
            point2 = x - alpha * g_hat
            descend_count = 0
            for round in range(0,num_round): 
                is_descend,bit_flipped = Comparison(point2,x)   # compare with x
                queries_count = queries_count + 1
                if is_descend == 1:
                    descend_count = descend_count + 1
            p = descend_count/num_round
            if p >= 0.5 + omega:
                return alpha,less_than_defalut,queries_count
    #else:
    #    alpha = last_step_size

    return alpha,less_than_defalut,queries_count


def SCOBO(Comparison,object_fcn,num_iterations,default_step_size,x0,r,m,d,s,fixed_flip_rate,line_search,warm_started):
    ''' 
    This function implements the SCOBO algorithm, as described 
    in our paper. 
    
    =============== INPUTS ================
    Comparison..................... comparison orcale
    num_iterations ................ number of iterations
    default_step_size ............. default step size
    x0 ............................ initial iterate
    r ............................. sampling radius
    kappa, delta_0,mu ............. oracle parameters
    m ............................. number of samples per iteration
    d ............................. dimension of problem
    s ............................. sparsity level
    fixed_flip_rate ............... ture if kappa==1, i.e., comparison orcale's flip rate is independent to |f(x)-f(y)|
    line_search ................... wheather linesearch for step size. if not, use default step size
    warm_started .................. wheather use warm start in linesearch
     
    =============== OUTPUTS ================
    regret ....................... vector of errors f(x_k) - min f
    tau_vec ...................... tau_vec(k) = fraction of flipped measurements at k-th iteration
    c_num_queries ................ cumulative number of queries.
    '''
    regret = np.zeros((num_iterations,1))
    tau_vec = np.zeros((num_iterations,1))
    linesearch_queries = np.zeros(num_iterations)
    x1 = np.squeeze(x0)
    x = np.copy(x1)
        
    
    Z = np.zeros((m,d))
    for i in range(0,m):
        temp = np.random.randn(1,d)
        Z[i,:] = temp/np.linalg.norm(temp)
    
    # start with default step size when using line search
    step_size = default_step_size
    
    if line_search:
        less_than_defalut_vec = np.zeros((num_iterations,1))  # not outputing this in current version
    
    for i in range(0,num_iterations):
        if fixed_flip_rate == 1:
            r = r * 0.99
        g_hat,tau = GradientEstimator(Comparison,x,Z,r,m,d,s)
        if line_search:
            if warm_started:
                if default_step_size >= 1e-4:
                    default_step_size = default_step_size*0.95
                else:
                    default_step_size = 1e-4
            #default_step_size = 1e-6
            step_size,less_than_defalut,queries_count = GetStepSize(Comparison,x,g_hat,step_size,default_step_size,warm_started)
            less_than_defalut_vec[i] = less_than_defalut
            linesearch_queries[i] = queries_count
            # print(queries_count)
        # print(step_size)
        x = x - step_size * g_hat
        regret[i] = object_fcn(x) # f(x_min) = 0
        tau_vec[i] = tau
        
        print('current regret at step', i+1, ':', regret[i])
        print('step_size:', step_size)
        #print('gradient norm:', np.linalg.norm(g_hat))
       
    c_num_queries = m*np.arange(start=0,stop = num_iterations,step = 1) + np.cumsum(linesearch_queries)
    #x_hat = x
    return x, regret,tau_vec,c_num_queries