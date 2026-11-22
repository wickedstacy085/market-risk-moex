"""Данные: индекс Мосбиржи (IMOEX) и ставка RUONIA."""
from pathlib import Path

import numpy as np
import pandas as pd


def load_index(path: str | Path = "data/imoex.csv") -> pd.Series:
    """Дневные лог-доходности IMOEX, %. Перерыв торгов 25.02–23.03.2022 даёт одну доходность за весь разрыв."""
    close = pd.read_csv(path, parse_dates=["date"]).set_index("date")["close"].sort_index()
    return (100 * np.log(close).diff()).dropna().rename("r")


def load_ruonia(path: str | Path = "data/ruonia.csv") -> pd.Series:
    """RUONIA, % годовых, по рабочим дням."""
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")["ruonia"].sort_index()
