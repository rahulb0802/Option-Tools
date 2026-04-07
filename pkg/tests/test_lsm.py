import pytest
import numpy as np
import pandas as pd
from final.lsm import lsm_price

def test_lsm():
    K = 1.10
    r = 0.06
    T = 3
    
    # stock price paths in paper
    S_matrix = [
     [1.00, 1.09, 1.08, 1.34],
     [1.00, 1.16, 1.26, 1.54],
     [1.00, 1.22, 1.07, 1.03],
     [1.00, 0.93, 0.97, 0.92],
     [1.00, 1.11, 1.56, 1.52],
     [1.00, 0.76, 0.77, 0.90],
     [1.00, 0.92, 0.84, 1.01],
     [1.00, 0.88, 1.22, 1.34],
    ]

    # test price against derived price
    price, _ = lsm_price(K=K, r=r, T=T, payoff='put', S=S_matrix)
    assert round(price, 4) == 0.1144