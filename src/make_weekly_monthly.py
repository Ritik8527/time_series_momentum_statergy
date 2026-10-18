"""
STEP 4 - Build WEEKLY and MONTHLY candles from the daily files.

How to combine daily candles into one bigger candle:
    Open   = open of the FIRST day      High  = highest High of all days
    Low    = lowest Low of all days     Close = close of the LAST day
    Volume = sum of volumes
We label each bar with the last real trading day inside it (a week ending on a holiday
Friday is labelled Thursday). The final bar may be unfinished; column "partial" says so.

Run:  python src/make_weekly_monthly.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "daily"


def resample_ohlc(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """daily has columns Date, Open, High, Low, Close, Volume. rule: 'W-FRI' (week) or 'M' (month)."""
    d = daily.sort_values("Date").reset_index(drop=True)
    key = d["Date"].dt.to_period(rule)
    out = d.groupby(key).agg(Date=("Date", "last"), Open=("Open", "first"), High=("High", "max"),
                             Low=("Low", "min"), Close=("Close", "last"), Volume=("Volume", "sum"))
    period_end = out.index.to_timestamp(how="end").normalize()
    last_weekday = (period_end + pd.offsets.BDay(1)) - pd.offsets.BDay(1)   # month ending on a Sunday -> Friday
    out["partial"] = last_weekday > d["Date"].max()
    return out.reset_index(drop=True)


def main():
    files = sorted(RAW.glob("*.csv"))
    if not files:
        raise SystemExit("No daily files found. Run download_daily.py first.")
    for sub in ("weekly", "monthly"):
        (ROOT / "data" / sub).mkdir(parents=True, exist_ok=True)
    for f in files:
        daily = pd.read_csv(f, parse_dates=["Date"])
        resample_ohlc(daily, "W-FRI").to_csv(ROOT / "data" / "weekly" / f.name, index=False)
        resample_ohlc(daily, "M").to_csv(ROOT / "data" / "monthly" / f.name, index=False)
    print(f"Built weekly and monthly files for {len(files)} stocks.")


if __name__ == "__main__":
    main()
