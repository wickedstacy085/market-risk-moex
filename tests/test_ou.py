import numpy as np
import pandas as pd
import pytest

from risk.ou import fit_ou, simulate_ou


def test_fit_recovers_parameters():
    t, x = simulate_ou(0.5, kappa=2, theta=0.5, sigma=0.2, T=200, n=200 * 252, paths=1, seed=7)
    est = fit_ou(pd.Series(x[0]), dt=1 / 252)
    assert est["kappa"] == pytest.approx(2, rel=0.15)
    assert est["theta"] == pytest.approx(0.5, rel=0.15)
    assert est["sigma"] == pytest.approx(0.2, rel=0.15)
    assert est["half_life_days"] == pytest.approx(np.log(2) / est["kappa"] * 252)


def test_non_stationary_raises():
    x = pd.Series(1.001 ** np.arange(1000))
    with pytest.raises(ValueError):
        fit_ou(x)


def test_simulation_moments():
    _, x = simulate_ou(0, kappa=2, theta=0.5, sigma=0.2, T=10, n=2000, paths=10_000, seed=42)
    assert x[:, -1].mean() == pytest.approx(0.5, abs=0.01)
    assert x[:, -1].var() == pytest.approx(0.2 ** 2 / 4, rel=0.05)
