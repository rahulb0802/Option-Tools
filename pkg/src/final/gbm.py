import numpy as np
from scipy.stats import norm

def geometric_brownian_motion(n_paths, n_steps, S0, mu, sigma, *, dt=1/252, seed=None, return_log=False, antithetic=False, mu_is=None, sigma_is=None):
    """
    Simulates a Geometric Brownian Motion (GBM) with n paths and t time steps. Parameters should be annualized. Option to seed generator.
    """
    M = n_paths
    N = n_steps
    # setting parameters (either true or importance sampling)
    sim_mu = mu_is if mu_is is not None else mu
    sim_sigma = sigma_is if sigma_is is not None else sigma

    rng = np.random.default_rng(seed=seed) # setting seed

    # antithetic logic
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("Number of paths must be even for antithetic sampling")
        Z_half = rng.standard_normal((int(M / 2), N))
        Z = np.concatenate([Z_half, -Z_half], axis=0) # get increment and its negative covariate
    else:
        Z = rng.standard_normal((M, N))
    
    # Getting log returns
    t1 = (sim_mu - 0.5 * sim_sigma**2) * dt
    t2 = sim_sigma * np.sqrt(dt) * Z
    return_factor = np.exp(t1 + t2)

    # converting to price movement
    prices = np.zeros((M, N+1))
    prices[:, 0] = S0
    prices[:, 1:] = S0 * np.cumprod(return_factor, axis=1)

    # importance sampling
    if mu_is is not None or sigma_is is not None:
        T = N * dt
        total_log_returns = np.log(prices[:, -1] / S0)

        # drift calcs
        drift_true = mu - 0.5 * sigma**2
        drift_sim = sim_mu - 0.5 * sigma**2

        # correction factors
        pdf_true = norm.pdf(total_log_returns, loc=drift_true * T, scale = sigma * np.sqrt(T))
        pdf_sim = norm.pdf(total_log_returns, loc=drift_sim * T, scale = sim_sigma * np.sqrt(T))

        weights = pdf_true / pdf_sim
    else:
        weights = np.ones(M)
    # normal log return calcs for accessibility
    if return_log:
        ret = np.log(prices[:, 1:] / prices[:, :-1])
    else:
        ret = prices[:, 1:] / prices[:, :-1] - 1
    return prices, ret, weights
