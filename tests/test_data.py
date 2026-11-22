import pandas as pd
from conftest import DATA

from risk.data import load_index, load_ruonia


def test_index_returns():
    r = load_index(DATA / "imoex.csv")
    assert r.index.is_unique and r.index.is_monotonic_increasing
    assert r.notna().all()
    # перерыв торгов 25.02–23.03.2022: одна доходность за весь разрыв
    gap = r.loc["2022-02-26":"2022-03-23"]
    assert gap.empty and "2022-03-24" in r.index
    assert r.min() < -30   # 24.02.2022


def test_ruonia():
    x = load_ruonia(DATA / "ruonia.csv")
    assert x.index.is_monotonic_increasing and x.between(0, 30).all()
