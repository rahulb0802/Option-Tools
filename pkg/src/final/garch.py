import numpy as np
from scipy.optimize import minimize
import pandas as pd

def get_garch_parameters(returns):
    """
    Fits a GARCH(1, 1) on returns, solving for parameters. Returns parameter estimates.
    """
    # express rets as percents for better convergence
    us = returns.values * 100
    # define objective (params and log likehood)
    def objective(params, us_data):
        alpha_0, alpha_1, beta_1 = params
        n = len(us_data)
        sigmas_sq = np.zeros(n, dtype=float)
        sigmas_sq[0] = np.var(us_data)
        

        for i in range(1, n):
            sigmas_sq[i] = alpha_0 + (alpha_1 * us_data[i-1]**2) + (beta_1 * sigmas_sq[i-1]) # variance estimator
        variances = pd.Series(sigmas_sq, index=returns.index)
        ll = np.sum(-np.log(variances.values) - us_data**2 / variances.values)
        return -ll # minimizing negative log-likelihood

    initial = [0.05, 0.1, 0.8] # initial guesses
    constraints = ({'type': 'ineq', 'fun': lambda x: 0.9999 - (x[1] + x[2])}) # a1 + b1 < 1
    bounds = ((1e-6, None), (0, 0.9999), (0, 0.9999)) # bounds of a0, a1, and b1

    res = minimize(objective, initial, args=(us,), bounds=bounds, constraints=constraints, method='SLSQP', tol=1e-15) # get params that minimize nll
    return res.x[0], res.x[1], res.x[2]

def get_garch_variances(returns, params):
    """
    Returns the calculated variances on the sample itself using the estimated parameters.
    """
    alpha_0, alpha_1, beta_1 = params
    us_data = returns.values * 100
    n = len(us_data)
    
    sigmas_sq = np.zeros(n)
    sigmas_sq[0] = np.var(us_data) 
    
    # same formula 
    for i in range(1, n):
        sigmas_sq[i] = alpha_0 + (alpha_1 * us_data[i-1]**2) + (beta_1 * sigmas_sq[i-1])
    
    return sigmas_sq
