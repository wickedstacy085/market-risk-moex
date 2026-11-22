"""Бэктест VaR: тесты Купика и Кристофферсена, светофор Базельского комитета."""
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2


def kupiec(n_exc: int, n: int, alpha: float = 0.99) -> tuple[float, float]:
    """Proportion of failures: совпадает ли доля пробоев с 1 − alpha. LR ~ χ²(1)."""
    p, x = 1 - alpha, n_exc
    phat = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen(exc: np.ndarray) -> tuple[float, float]:
    """Независимость пробоев: зависит ли вероятность пробоя от того, был ли пробой вчера. LR ~ χ²(1)."""
    e = np.asarray(exc, int)
    prev, cur = e[:-1], e[1:]
    n00 = np.sum((prev == 0) & (cur == 0)); n01 = np.sum((prev == 0) & (cur == 1))
    n10 = np.sum((prev == 1) & (cur == 0)); n11 = np.sum((prev == 1) & (cur == 1))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return float(lr), float(chi2.sf(lr, 1))


def traffic_light(n_exc_250: int) -> str:
    """Зоны Базельского комитета для VaR 99% на 250 днях: 0–4 / 5–9 / 10+ пробоев."""
    return "зелёная" if n_exc_250 <= 4 else ("жёлтая" if n_exc_250 <= 9 else "красная")


def exceptions(r: pd.Series, var_df: pd.DataFrame) -> pd.Series:
    """1, если потеря дня больше прогноза VaR; только дни с прогнозом."""
    ok = var_df["var"].notna()
    return (r[ok] < -var_df.loc[ok, "var"]).astype(int)


def backtest_table(r: pd.Series, var_df: pd.DataFrame, alpha: float = 0.99) -> dict:
    e = exceptions(r, var_df)
    n, x = len(e), int(e.sum())
    hit = e[e == 1].index
    return {"n": n, "exceptions": x, "expected": n * (1 - alpha), "rate": x / n,
            "kupiec_p": kupiec(x, n, alpha)[1], "christoffersen_p": christoffersen(e.to_numpy())[1],
            "avg_loss_on_exc": float(-r[hit].mean()) if x else np.nan,
            "avg_var_on_exc": float(var_df.loc[hit, "var"].mean()) if x else np.nan,
            "avg_es_on_exc": float(var_df.loc[hit, "es"].mean()) if x else np.nan}


def zones_by_year(r: pd.Series, var_df: pd.DataFrame) -> pd.DataFrame:
    e = exceptions(r, var_df)
    t = e.groupby(e.index.year).agg(days="size", exceptions="sum")
    t["zone"] = t["exceptions"].map(traffic_light)
    return t
