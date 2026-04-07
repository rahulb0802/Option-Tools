# Options Pricing Final Project

A collection of option pricing tools built from scratch for FM 5151. The project covers Black-Scholes analytics, GBM simulation, GARCH(1,1) volatility estimation, and American option pricing via the Longstaff-Schwartz (LSM) algorithm, along with variance reduction techniques (antithetic sampling and importance sampling).

---

## Project Structure

```
final_files/
├── data/
│   ├── options.parquet          # raw options data
│   ├── options_final.parquet    # IV-solved options data
│   └── stock.parquet            # historical stock price data
├── notebooks/
│   ├── part2.ipynb              # Delta and gamma hedging implementation
│   ├── part3.ipynb              # LSM pricing & variance reduction
│   ├── part4.ipynb              # GARCH(1,1) volatility estimation
│   └── part5.ipynb              # IV root finding + finite difference approx
└── pkg/
├── pyproject.toml               # package config
    └── src/final/
        ├── bs_analytic.py       # Black-Scholes price & Greeks
        ├── gbm.py               # GBM path simulator
        ├── lsm.py               # Longstaff-Schwartz American option pricer
        └── garch.py             # GARCH(1,1) parameter estimation
    └── tests/
        ├── test_analytic.py  # Black-Scholes tests
        ├── test_gbm.py       # GBM tests
        ├── test_lsm.py       # LSM tests        
```

---

## Implemented Features

### Black-Scholes (`bs_analytic.py`)
Analytical closed-form pricing for European calls and puts, plus three Greeks: delta, gamma, and theta. Supports continuous dividend yield `q`.

### GBM Simulation (`gbm.py`)
Simulates stock price paths under the standard log-normal diffusion model. Key features:
- Antithetic variates for variance reduction
- Importance sampling (shifts the drift to oversample the tail region)
- Returns prices, simple/log returns, and likelihood ratio weights in one call

### Longstaff-Schwartz (`lsm.py`)
Prices American options (calls and puts) by backward induction over simulated paths. Uses OLS regression (quadratic basis) on in-the-money paths to estimate continuation values at each step. Can be called with pregenerated price paths or will simulate internally via GBM. Antithetic and importance sampling are both supported in this function.

### GARCH(1,1) (`garch.py`)
Fits a GARCH(1,1) model to a return series by maximizing the log-likelihood via `scipy.optimize.minimize` (SLSQP solver). Returns the three parameters `(α₀, α₁, β₁)` and can reconstruct the conditional variance series for the full sample.

---

## Installation

The pricing library is packaged under `pkg/`. Install it in editable mode from the repo root:

```bash
cd pkg
pip install -e .
```

## Running Tests

```bash
cd pkg
pip install -e ".[test]"
pytest --cov=final tests/
```

The test suite checks:
- BS price and Greeks against known analytical values (ITM/OTM, calls/puts)
- GBM output shape, finite values, and convergence of the simulated expected return
- LSM price against the example paths from the original Longstaff-Schwartz (2001) paper

---

## Notebooks

Each notebook is self-contained and imports from the `final` package after installation.

| Notebook | Content |
|---|---|
| `part2.ipynb` | Delta and Gamma Hedging Implementation on a short put option |
| `part3.ipynb` | LSM American option pricing and variance reduction analysis |
| `part4.ipynb` | GARCH(1,1) fitting on real stock return data |
| `part5.ipynb` | Root-finding to solve for implied volatility and finite diff approximations for Greeks |
