"""Процесс Орнштейна–Уленбека dX = κ(θ − X)dt + σdW (модель ставок Васичека)."""
import numpy as np
import pandas as pd


def fit_ou(x: pd.Series, dt: float = 1 / 252) -> dict:
    """Оценка по точной дискретизации: X_{t+1} = a + b·X_t + ε, b = e^{−κΔt}, θ = a/(1−b),
    σ = sd(ε)·√(2κ / (1 − b²)). При b ≥ 1 ряд не возвращается к среднему — OU к нему не применим."""
    v = np.asarray(x, float)
    y, z = v[1:], v[:-1]
    b, a = np.polyfit(z, y, 1)
    if not 0 < b < 1:
        raise ValueError(f"Оценка b = {b:.5f}: ряд не возвращается к среднему, модель OU не подходит")
    kappa = -np.log(b) / dt
    resid = y - (a + b * z)
    sigma = resid.std(ddof=2) * np.sqrt(2 * kappa / (1 - b ** 2))
    return {"a": a, "b": b, "kappa": kappa, "theta": a / (1 - b), "sigma": sigma,
            "half_life_days": np.log(2) / kappa / dt}


def simulate_ou(x0: float, kappa: float, theta: float, sigma: float, T: float = 1.0, n: int = 250,
                paths: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Точная дискретизация: каждый шаг считается от предыдущего значения."""
    rng = np.random.default_rng(seed)
    dt = T / n
    a = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1 - a ** 2) / (2 * kappa))
    x = np.empty((paths, n + 1)); x[:, 0] = x0
    for j in range(1, n + 1):
        x[:, j] = theta + (x[:, j - 1] - theta) * a + sd * rng.standard_normal(paths)
    return np.linspace(0, T, n + 1), x
