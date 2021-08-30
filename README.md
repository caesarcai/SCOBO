# SCOBO
This Python repo is for a comparison orcale based optimization algorithm introduced in [1], which is coined *S*parsity-aware *Co*mparison-*B*ased *O*ptimization (SCOBO).

## Input Description
1. Comparison..................... comparison orcale
1. num_iterations ................ number of iterations
1. default_step_size ............. default step size
1. x0 ............................ initial iterate
1. r ............................. sampling radius
1. kappa, delta_0,mu ............. oracle parameters
1. m ............................. number of samples per iteration
1. d ............................. dimension of problem
1. s ............................. sparsity level
1. fixed_flip_rate ............... ture if kappa==1, i.e., comparison orcale's flip rate is independent to |f(x)-f(y)|
1. line_search ................... wheather linesearch for step size. if not, use default step size
1. warm_started .................. wheather use warm start in linesearch
     
## Output Description
1. regret ....................... vector of errors f(x_k) - min f
1. tau_vec ...................... tau_vec(k) = fraction of flipped measurements at k-th iteration
1. c_num_queries ................ cumulative number of queries.

## Demo
Run `test_sobo.py` for demo. Therein, we include four test problems. More details about the test problems can be found in the paper. 




## Reference
[1] H.Q. Cai, D. Mckenzie, W. Yin, and Z. Zhang. A One-bit, Comparison-Based Gradient Estimator. *arXiv: 2010.02479*.
