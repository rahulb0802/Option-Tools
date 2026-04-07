import numpy as np
from .gbm import geometric_brownian_motion
from scipy.stats import norm

def lsm_price(K, r, T, payoff='call', S=None, *, S0=None, mu=None, sigma=None, m=None, n=None, seed=42, antithetic=False, importance=False):
    """
    Function that uses LSM to generate price and variance estimates for American options. Includes antithetic and importance sampling options.
    """
    # initialize for later check
    weights = None
    dt = None

    if S is None:
        if any(param is None for param in [S0, mu, sigma, m]):
            raise ValueError("If S is not provided, you must provide the GBM params for generation)")
        dt = T / n
        mu_is = None
        sigma_is = None
        if importance:
            drift = mu - 0.5 * sigma**2
            diffusion = sigma
            distribution = norm(loc=drift * T, scale=diffusion * np.sqrt(T))

            # we need the other tail for calls
            if payoff == 'call':
                b_outside = distribution.ppf(0.99)
            else:
                b_outside = distribution.ppf(0.01)

            b_bound = np.log(K / S0)
            target_return = (b_outside + b_bound) / 2
            target_drift = target_return / T
            mu_is = target_drift + 0.5 * sigma**2 # (mu_is - 0.5*sigma^2) = target_drift (solving for mu that gives target drift)

        S, _, weights = geometric_brownian_motion(m, n, S0, mu, sigma, dt=dt, seed=seed, antithetic=antithetic, mu_is=mu_is, sigma_is=sigma_is)
    S = np.asarray(S)
    m, n = S.shape[0], S.shape[1] - 1 # make sure these follow shape of whatever S is

    if dt is None:
        dt = T / n

    if weights is None:
        weights = np.ones(m)

    time_index = np.arange(n + 1) * dt
    df = np.exp(-r * time_index)

    # payoff funcs
    if payoff == 'call':
        payoff_func = lambda S: np.maximum(S - K, 0)
    else:
        payoff_func = lambda S: np.maximum(K - S, 0)

    cf = payoff_func(S[:, -1])
    cft = np.full(m, n, dtype=int) # vector of size M set to Ns

    for j in range(n - 1, 0, -1):
        po = payoff_func(S[:, j])
        itm = po > 0
        itm_indices = np.where(itm)[0]

        # skip current time step if we have less than 2 indices
        if len(itm_indices) < 2:
            continue

        Y = cf[itm_indices] * df[cft[itm_indices]] / df[j]
        X = S[:, j][itm_indices]

        # regression
        X_MAT = np.stack([X**0, X**1, X**2], axis=1)
        beta_hat = np.linalg.solve(X_MAT.T @ X_MAT, X_MAT.T @ Y)
        Y_FIT = X_MAT @ beta_hat

        exercise = po[itm_indices] > Y_FIT
        new_indices = itm_indices[exercise]

        cf[new_indices] = po[new_indices]
        cft[new_indices] = j
    
    weighted_dcf = cf * df[cft] * weights # apply weights from importance sampling if present
    price = np.mean(weighted_dcf)
    if antithetic:
        # split into normal and antithetic halves (we concatenated in gbm sim so this should work)
        mid = m // 2
        paths_1 = weighted_dcf[:mid]
        paths_2 = weighted_dcf[mid:]
        
        # average pairs and find variance of those averages
        pair_averages = (paths_1 + paths_2) / 2
        var_pairs = np.var(pair_averages)
        
        # SE = sqrt(Var(pairs) / (num pairs)) since we have n/2 "samples" now
        var = 2 * var_pairs
    else:
        var = np.var(weighted_dcf)

    return price, var

