"""Скользящий однодневный VaR и ES четырьмя методами. Потери — положительные числа, % от позиции.

Прогноз на дату t строится только по доходностям до t−1 включительно."""
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm


def ewma_sigma(r: pd.Series, lam: float = 0.94, init_window: int = 250) -> pd.Series:
    """RiskMetrics: σ²_t = λ·σ²_{t−1} + (1−λ)·r²_{t−1}; σ²_0 — выборочная дисперсия первого окна.
    σ_t — прогноз на день t, известный в конце дня t−1."""
    x = r.to_numpy(float)
    s2 = np.empty(len(x))
    s2[0] = x[:init_window].var(ddof=1)
    for t in range(1, len(x)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * x[t - 1] ** 2
    return pd.Series(np.sqrt(s2), index=r.index)


def _hist(h: np.ndarray, alpha: float, alpha_es: float) -> tuple[float, float]:
    var = -np.quantile(h, 1 - alpha)
    q_es = np.quantile(h, 1 - alpha_es)
    return var, -h[h <= q_es].mean()


def rolling_var(r: pd.Series, method: Literal["hist", "normal", "ewma", "fhs"], alpha: float = 0.99,
                window: int = 250, lam: float = 0.94, alpha_es: float = 0.975) -> pd.DataFrame:
    """DataFrame с колонками var и es; первые `window` дат — NaN (нет истории)."""
    x = r.to_numpy(float)
    out = np.full((len(x), 2), np.nan)
    z, z_es = norm.ppf(alpha), norm.ppf(alpha_es)
    if method in ("ewma", "fhs"):
        sig = ewma_sigma(r, lam, window).to_numpy()
    for t in range(window, len(x)):
        h = x[t - window:t]
        if method == "hist":
            out[t] = _hist(h, alpha, alpha_es)
        elif method == "normal":
            mu, sd = h.mean(), h.std(ddof=1)
            out[t] = (-mu + sd * z, -mu + sd * norm.pdf(z_es) / (1 - alpha_es))
        elif method == "ewma":
            out[t] = (sig[t] * z, sig[t] * norm.pdf(z_es) / (1 - alpha_es))
        elif method == "fhs":
            # filtered historical simulation: исторические стандартизованные остатки × текущая волатильность
            std_resid = h / sig[t - window:t]
            v, e = _hist(std_resid, alpha, alpha_es)
            out[t] = (v * sig[t], e * sig[t])
        else:
            raise ValueError(f"Неизвестный метод: {method}")
    return pd.DataFrame(out, index=r.index, columns=["var", "es"])
