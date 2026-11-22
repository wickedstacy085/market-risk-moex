import numpy as np
import pytest

from risk.backtest import christoffersen, kupiec, traffic_light


def test_kupiec_expected_count():
    lr, p = kupiec(10, 1000, 0.99)
    assert p == pytest.approx(1, abs=1e-6)


def test_kupiec_zero_exceptions_finite():
    lr, p = kupiec(0, 1000, 0.99)
    assert np.isfinite(lr) and lr > 0


def test_kupiec_too_many():
    assert kupiec(30, 1000, 0.99)[1] < 0.01


def test_christoffersen_cluster_vs_scattered():
    clustered = np.zeros(1000, int); clustered[500:510] = 1
    assert christoffersen(clustered)[1] < 0.01
    scattered = np.zeros(1000, int); scattered[::100] = 1
    assert christoffersen(scattered)[1] > 0.05


def test_christoffersen_no_pairs_no_nan():
    e = np.zeros(500, int); e[100] = 1
    lr, p = christoffersen(e)
    assert np.isfinite(lr) and np.isfinite(p)


@pytest.mark.parametrize("n,zone", [(0, "зелёная"), (4, "зелёная"), (5, "жёлтая"), (9, "жёлтая"), (10, "красная")])
def test_traffic_light(n, zone):
    assert traffic_light(n) == zone
