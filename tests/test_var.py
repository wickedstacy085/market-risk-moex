import numpy as np
import pandas as pd
import pytest

from risk.var import rolling_var

rng = np.random.default_rng(1)


def _series(x):
    return pd.Series(x, index=pd.bdate_range("2010-01-01", periods=len(x)))


def test_normal_var_and_es_on_standard_normal():
    r = _series(rng.standard_normal(200_000))
    v = rolling_var(r, "normal", window=199_000).dropna().iloc[-1]
    assert v["var"] == pytest.approx(2.326, abs=0.03)
    assert v["es"] == pytest.approx(2.338, abs=0.03)


def test_hist_var_is_empirical_quantile():
    x = np.arange(1, 251, dtype=float) * -0.01    # потери 0,01 … 2,5
    r = _series(np.r_[x, 0.0])
    v = rolling_var(r, "hist", alpha=0.99, window=250).iloc[-1]
    assert v["var"] == pytest.approx(-np.quantile(x, 0.01))


@pytest.mark.parametrize("method", ["hist", "normal", "ewma", "fhs"])
def test_no_look_ahead(method):
    r = _series(rng.standard_normal(400))
    a = rolling_var(r, method, window=250)
    r2 = r.copy(); r2.iloc[300] = -50
    b = rolling_var(r2, method, window=250)
    # прогноз на дату 300 построен по данным до 299 и не должен измениться
    assert a.iloc[300]["var"] == pytest.approx(b.iloc[300]["var"])
    assert b.iloc[301]["var"] >= a.iloc[301]["var"]


def test_ewma_reacts_to_shock():
    r = _series(np.r_[rng.normal(0, 1, 300), [-10.0], rng.normal(0, 1, 5)])
    v = rolling_var(r, "ewma", window=250)
    assert v.iloc[301]["var"] > 1.5 * v.iloc[300]["var"]
