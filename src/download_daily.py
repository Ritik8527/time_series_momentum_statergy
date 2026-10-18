"""
STEP 3 - Download DAILY candles for every F&O stock.

Facts you must know about the Angel One historical API:
  * It gives minute candles up to ONE_DAY candles. It has NO weekly or monthly interval,
    so weekly/monthly bars are built from daily bars in step 4.
  * One call returns a limited number of candles (about 2000 days for ONE_DAY), so we cut the
    date range into chunks.
  * Calls are rate-limited, so we pause between calls and retry politely on errors.
  * Prices are RAW (not adjusted for splits/bonus). Step 5 looks for those.

Run a tiny test first:   python src/download_daily.py --limit 3
Then everything:         python src/download_daily.py
Update later:            python src/download_daily.py     (it only fetches new days)
"""
import argparse
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "meta"
RAW = ROOT / "data" / "raw" / "daily"

CHUNK_DAYS = 1500        # stay safely under the ~2000-candle limit per call
PAUSE = 0.4              # seconds between calls (rate limit)
OVERLAP_DAYS = 5         # when updating, re-fetch the last few days to repair revisions


def safe_name(symbol: str) -> str:
    """M&M -> M_M so the file name is valid on every system."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", symbol)


def date_windows(start: datetime, end: datetime, chunk_days: int = CHUNK_DAYS):
    """Cut [start, end] into consecutive windows of at most chunk_days days."""
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def parse_candles(rows) -> pd.DataFrame:
    """Angel One rows look like [timestamp, open, high, low, close, volume]."""
    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["ts"]).dt.tz_localize(None).dt.normalize()
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


def call_with_retry(client, params: dict, tries: int = 4):
    last = None
    for k in range(tries):
        try:
            res = client.getCandleData(params)
        except Exception as e:                       # network hiccup, expired session ...
            res = {"status": False, "message": repr(e)}
        if res and res.get("status"):
            return res.get("data") or []
        last = res.get("message") if isinstance(res, dict) else res
        if k < tries - 1:
            print(f"    problem ({last}); retrying ...")
            time.sleep(PAUSE * 2 ** (k + 1))
    raise RuntimeError(f"giving up after {tries} tries: {last}")


def fetch_history(client, token, start: datetime, end: datetime, exchange: str = "NSE") -> pd.DataFrame:
    frames = []
    for a, b in date_windows(start, end):
        params = {
            "exchange": exchange,
            "symboltoken": str(token),
            "interval": "ONE_DAY",
            "fromdate": a.strftime("%Y-%m-%d 09:15"),
            "todate": b.strftime("%Y-%m-%d 15:30"),
        }
        rows = call_with_retry(client, params)
        if rows:
            frames.append(parse_candles(rows))
        time.sleep(PAUSE)
    if not frames:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    return pd.concat(frames).drop_duplicates("Date").sort_values("Date").reset_index(drop=True)


def update_symbol(client, symbol: str, token, start: datetime, end: datetime, force: bool = False) -> str:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{safe_name(symbol)}.csv"
    old = None
    if path.exists() and not force:
        old = pd.read_csv(path, parse_dates=["Date"])
        if len(old):
            start = max(start, old["Date"].max().to_pydatetime() - timedelta(days=OVERLAP_DAYS))
    if start > end:
        return "already up to date"
    new = fetch_history(client, token, start, end)
    if old is not None and len(old):
        new = pd.concat([old, new]).drop_duplicates("Date", keep="last").sort_values("Date")
    new.to_csv(path, index=False)
    return f"{len(new)} rows, {new['Date'].min():%Y-%m-%d} to {new['Date'].max():%Y-%m-%d}" if len(new) else "NO DATA"


def default_end() -> datetime:
    """Today only after the market has closed (16:00), otherwise yesterday (no half-finished candle)."""
    now = datetime.now()
    day = now if now.hour >= 16 else now - timedelta(days=1)
    return datetime(day.year, day.month, day.day)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--symbols", default=None, help="comma separated, e.g. RELIANCE,TCS")
    ap.add_argument("--limit", type=int, default=None, help="only the first N stocks (testing)")
    ap.add_argument("--force", action="store_true", help="re-download from --start, ignore saved files")
    args = ap.parse_args()

    uni = pd.read_csv(META / "fno_universe.csv", dtype={"token": str})
    if args.symbols:
        wanted = {s.strip().upper() for s in args.symbols.split(",")}
        uni = uni[uni["symbol"].isin(wanted)]
    if args.limit:
        uni = uni.head(args.limit)

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else default_end()

    from angel_client import get_client, logout      # imported here so tests need no SmartApi
    client = get_client()
    failed = []
    try:
        for i, row in enumerate(uni.itertuples(index=False), 1):
            print(f"[{i}/{len(uni)}] {row.symbol} ...", end=" ", flush=True)
            try:
                print(update_symbol(client, row.symbol, row.token, start, end, args.force))
            except Exception as e:
                print("FAILED:", e)
                failed.append(f"{row.symbol}\t{e}")
    finally:
        logout(client)
    META.mkdir(parents=True, exist_ok=True)
    (META / "failed_downloads.txt").write_text("\n".join(failed), encoding="utf-8")
    print(f"\nDone. {len(uni) - len(failed)} ok, {len(failed)} failed (see data/meta/failed_downloads.txt)")


if __name__ == "__main__":
    main()
