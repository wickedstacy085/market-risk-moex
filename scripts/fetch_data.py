"""Обновляет data/imoex.csv (MOEX ISS) и data/ruonia.csv (ЦБ РФ). Нужен только для обновления данных."""
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

try:  
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

DATA = Path(__file__).resolve().parents[1] / "data"
ISS = ("https://iss.moex.com/iss/history/engines/stock/markets/index/securities/IMOEX.json"
       "?from=2013-01-01&iss.meta=off&history.columns=TRADEDATE,CLOSE&start={start}")
CBR = ("https://www.cbr.ru/hd_base/ruonia/dynamics/?UniDbQuery.Posted=True"
       "&UniDbQuery.From=01.01.2014&UniDbQuery.To={to}")


def fetch_imoex() -> pd.DataFrame:
    rows, start = [], 0
    while True:
        with urllib.request.urlopen(ISS.format(start=start), timeout=60) as f:
            page = json.load(f)["history"]["data"]
        if not page:
            break
        rows += page
        start += len(page)
    df = pd.DataFrame(rows, columns=["date", "close"]).dropna().drop_duplicates("date")
    return df.sort_values("date")


def fetch_ruonia() -> pd.DataFrame:
    to = pd.Timestamp.today().strftime("%d.%m.%Y")
    req = urllib.request.Request(CBR.format(to=to), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as f:
        html = f.read().decode("utf-8")
    t = pd.read_html(io.StringIO(html), decimal=",", thousands=" ")[0]
    df = pd.DataFrame({"date": pd.to_datetime(t.iloc[:, 0], format="%d.%m.%Y").dt.strftime("%Y-%m-%d"),
                       "ruonia": t.iloc[:, 1]})
    return df.sort_values("date")


if __name__ == "__main__":
    fetch_imoex().to_csv(DATA / "imoex.csv", index=False)
    fetch_ruonia().to_csv(DATA / "ruonia.csv", index=False)
    print("ok")
