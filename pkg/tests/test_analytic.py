import pytest
import quanttools.analytic as qa

def test_bs_call_ITM():
    S = 75.5
    K = 50.25
    r = 0.10
    q = 0.02
    sigma = 0.35
    T = 2.5

    price = qa.bs_price(S, K, T, r, sigma, q, is_call=True)
    delta = qa.bs_delta(S, K, T, r, sigma, q, is_call=True)
    gamma = qa.bs_gamma(S, K, T, r, sigma, q)
    theta = qa.bs_theta(S, K, T, r, sigma, q, is_call=True)

    assert pytest.approx(price, 1e-9) == 34.6578940888037
    assert pytest.approx(delta, 1e-9) == 0.870607763233823
    assert pytest.approx(gamma, 1e-9) == 0.00353504880571565
    assert pytest.approx(theta, 1e-9) == -0.0082929028103623 * 365


def test_bs_call_OTM():
    S = 35.4
    K = 55.25
    r = 0.08
    q = 0.04
    sigma = 0.25
    T = 1.5

    price = qa.bs_price(S, K, T, r, sigma, q, is_call=True)
    delta = qa.bs_delta(S, K, T, r, sigma, q, is_call=True)
    gamma = qa.bs_gamma(S, K, T, r, sigma, q)
    theta = qa.bs_theta(S, K, T, r, sigma, q, is_call=True)

    assert pytest.approx(price, 1e-9) == 0.61084957151139
    assert pytest.approx(delta, 1e-9) == 0.126778632636872
    assert pytest.approx(gamma, 1e-9) == 0.0188280918577903
    assert pytest.approx(theta, 1e-9) == -0.00237803339824326 * 365

def test_bs_put_ITM():
    S = 40.15
    K = 65.25
    r = 0.08
    q = 0.03
    sigma = 0.25
    T = 2.33

    price = qa.bs_price(S, K, T, r, sigma, q, is_call=False)
    delta = qa.bs_delta(S, K, T, r, sigma, q, is_call=False)
    gamma = qa.bs_gamma(S, K, T, r, sigma, q)
    theta = qa.bs_theta(S, K, T, r, sigma, q, is_call=False)

    assert pytest.approx(price, 1e-9) == 18.2202245870577
    assert pytest.approx(delta, 1e-9) == -0.728506968428777
    assert pytest.approx(gamma, 1e-9) == 0.0179615806241197
    assert pytest.approx(theta, 1e-9) == 0.00552128343262734 * 365

def test_bs_put_OTM():
    S = 55.5
    K = 45.5
    r = 0.06
    q = 0.02
    sigma = 0.40
    T = 1.75

    price = qa.bs_price(S, K, T, r, sigma, q, is_call=False)
    delta = qa.bs_delta(S, K, T, r, sigma, q, is_call=False)
    gamma = qa.bs_gamma(S, K, T, r, sigma, q)
    theta = qa.bs_theta(S, K, T, r, sigma, q, is_call=False)

    assert pytest.approx(price, 1e-9) == 4.75915233585737
    assert pytest.approx(delta, 1e-9) == -0.212398368063912
    assert pytest.approx(gamma, 1e-9) == 0.00973454908377432
    assert pytest.approx(theta, 1e-9) == -0.00449784676156258 * 365

