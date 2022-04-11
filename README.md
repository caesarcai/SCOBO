# SCOBO
This Python repo is for a comparison orcale based optimization algorithm introduced in <a href=https://arxiv.org/abs/2010.02479>[1]</a>, which is coined *S*parsity-aware *Co*mparison-*B*ased *O*ptimization (SCOBO).

###### To display math symbols properly, one may have to install a MathJax plugin. For example, [MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima?hl=en).

## The Problem
We aim to minimize an objective function $f:\mathbb{R}^d \rightarrow \mathbb{R}$ via merely its noisy comparison orcales $\mathcal{C}_f(\cdot,\cdot):\mathbb{R}^d \times \mathbb{R}^d \rightarrow $\{$-1,+1$\}, where 
$$\mathbb{P}[\mathcal{C}_f(x,y)=\mathrm{sign}(f(y)-f(x))]=\theta(|f(y)-f(x)|)$$
with some monotonically increasing $\theta$ that $\theta(0)\geq 0.5$. In words, there is chance that the comparison orcale may return a wrong sign, but this chance is always less than $50$%. In addition, when comparing two more distinct points, the chance of getting a wrong sign is no worse than comparing two less distinct points.


## Syntex
```
x, regret, tau_vec , c_num_queries = SCOBO(comparison, obj_fcn, num_iterations, default_step_size, x0, r, m, d, s, fixed_flip_rate, line_search, warm_started)
```

## Input Description
1. comparison : handle of the comparison orcale
1. object_fcn : objective function, this is for recording regret only. not used for solving problem
1. num_iterations : number of iterations
1. default_step_size : default step size
1. x0 : initial iterate
1. r : sampling radius
1. m : number of samples per iteration
1. d : dimension of problem
1. s : sparsity level
1. fixed_flip_rate : ture if kappa==1, i.e., comparison orcale's flip rate is independent to |f(x)-f(y)|; otherwise false
1. line_search : wheather linesearch for step size. if not, use default step size
1. warm_started : wheather use warm start in linesearch
     
## Output Description
1. x : estimated optimum point 
1. regret : vector of errors f(x_k) - min f
1. tau_vec : tau_vec(k) = fraction of flipped measurements at k-th iteration
1. c_num_queries : cumulative number of queries.

## Demo
Run `test_sobo.py` for demo. Therein, we include four test problems. More details about the test problems can be found in the paper. 


## Reference
[1] HanQin Cai, Daniel Mckenzie, Wotao Yin, and Zhenliang Zhang. <a href=https://doi.org/10.1016/j.acha.2022.03.003>A One-bit, Comparison-Based Gradient Estimator</a>. *Applied and Computational Harmonic Analysis*, 60: 242-266, 2022.
