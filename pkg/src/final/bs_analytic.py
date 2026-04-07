import numpy as np
from scipy.stats import norm

def bs_price(S, K, T, r, sigma, q=0, is_call=True):
    """
    Calculate Black-Scholes European option price using analytical formula.
    """
    d1 = (np.log(S/K) + T * (r - q + 0.5 * (sigma**2))) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    if is_call:
        return call
    else:
        return put

def bs_delta(S, K, T, r, sigma, q=0, is_call=True):
    """
    Calculates Black-Scholes European option delta using analytical formula.
    """
    d1 = (np.log(S/K) + T * (r - q + 0.5 * (sigma**2))) / (sigma * np.sqrt(T))
    call_delta = np.exp(-q * T) * norm.cdf(d1)
    put_delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
    if is_call:
        return call_delta
    else:
        return put_delta

def bs_gamma(S, K, T, r, sigma, q=0):
    """
    Calculates Black-Scholes European option gamma using analytical formula.
    """
    d1 = (np.log(S/K) + T * (r - q + 0.5 * (sigma**2))) / (sigma * np.sqrt(T))
    gamma = (norm.pdf(d1) * np.exp(-q * T)) / (S * sigma * np.sqrt(T))
    return gamma

def bs_theta(S, K, T, r, sigma, q=0, is_call=True):
    """
    Calculates Black-Scholes European option theta using analytical formula.
    """
    d1 = (np.log(S/K) + T * (r - q + 0.5 * (sigma**2))) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    call_theta = (-S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) + (q * S * norm.cdf(d1) * np.exp(-q * T)) - (r * K * np.exp(-r * T) * norm.cdf(d2))
    put_theta = (-S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) - (q * S * norm.cdf(-d1) * np.exp(-q * T)) + (r * K * np.exp(-r * T) * norm.cdf(-d2))
    if is_call:
        return call_theta
    else:
        return put_theta

