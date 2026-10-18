"""TSMOM step 1 - load daily close prices of all downloaded stocks into one table."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]     # the angel_pipeline folder
DAILY = ROOT / "data" / "raw" / "daily"        # where download_daily.py saved the files
IN_SAMPLE_END = "2023-12-31"                   # data after this date is never loaded


def load_close_prices(folder: Path = DAILY, end: str = IN_SAMPLE_END) -> pd.DataFrame:
    """Return a table: rows = dates, columns = stock symbols, values = close price."""
    series = {}                                          # symbol -> its Close prices
    for f in sorted(folder.glob("*.csv")):               # every csv file in the folder
        df = pd.read_csv(f, parse_dates=["Date"])        # read, and turn Date text into real dates
        s = df.set_index("Date")["Close"].sort_index()   # Date becomes the row label, keep Close only
        s = s[~s.index.duplicated(keep="last")]          # if a date appears twice, keep the last
        series[f.stem] = s.loc[:end]                     # f.stem = file name without .csv; cut at end date
    return pd.DataFrame(series)                          # dict of series -> table, aligned on dates


def clean_universe(prices: pd.DataFrame, min_days: int = 260):
    """Drop NSE test instruments and stocks with too little history.
    Returns (clean_prices, dropped) where dropped is a dict of reason -> list of names."""
    test = [c for c in prices.columns if "NSETEST" in c.upper()]       # NSE dummy symbols, not real stocks
    rest = prices.drop(columns=test)
    too_short = list(rest.columns[rest.count() < min_days])            # can never produce a 12-month signal
    clean = rest.drop(columns=too_short)
    return clean, {"test_symbols": test, "too_short": too_short}


def big_moves(prices: pd.DataFrame, threshold: float = -0.25) -> pd.DataFrame:
    """List every single-day move at or below `threshold` (e.g. -0.25 = -25%), with its date."""
    ret = prices.pct_change(fill_method=None)                          # daily returns, no gap filling
    long = ret.reset_index().melt(id_vars="Date", var_name="Symbol", value_name="Return").dropna()
    hits = long[long["Return"] <= threshold]
    return hits.sort_values("Return").reset_index(drop=True)


if __name__ == "__main__":
    prices = load_close_prices()
    print("loaded (days, stocks):", prices.shape)

    prices, dropped = clean_universe(prices)
    print("dropped test symbols :", len(dropped["test_symbols"]))
    print("dropped too short    :", dropped["too_short"])
    print("clean (days, stocks) :", prices.shape)
    print("first date:", prices.index.min().date(), " last date:", prices.index.max().date())

    hits = big_moves(prices, -0.25)
    print(f"\n{len(hits)} one-day drops of -25% or worse:")
    print(hits.to_string(index=False))