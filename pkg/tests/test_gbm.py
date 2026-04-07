import pytest
from final.gbm import geometric_brownian_motion
import numpy as np

def test_geometric_brownian_motion():
    S0 = 100
    mu = 0.05
    sigma = 0.2
    steps = 100
    paths = 100000

    # testing shape and whether all vals are valid
    prices, returns, _ = geometric_brownian_motion(n_paths=paths, n_steps=steps, S0=S0, mu=mu, sigma=sigma, seed=42, return_log=False)
    assert prices.shape == (paths, steps + 1)
    assert returns.shape == (paths, steps)
    assert np.isfinite(paths).all()

    # verifying simulated expected return
    expected_return = (1 + (mu/252))**steps
    assert pytest.approx(np.mean(prices[:, -1] / prices[:, 0]), 1e-3) == expected_return

    # sigma = 0 (deterministic)
    prices_deter, returns_deter, _ = geometric_brownian_motion(n_paths=paths, n_steps=steps, S0=S0, mu=mu, sigma=0, seed=42)
    expected_deter = S0 * np.exp(mu/252 * np.arange(steps + 1)) 
    for i in range(paths):
        assert np.allclose(prices_deter[i], expected_deter)