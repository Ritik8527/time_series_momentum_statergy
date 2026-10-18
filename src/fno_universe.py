"""
STEP 2 - Find every stock that is in the F&O segment.

Idea: Angel One publishes a public file ("instrument master") listing every tradable
instrument. A stock is "F&O listed" if it has stock futures (instrument type FUTSTK) on NFO.
For each such stock we then look up its normal cash-market (NSE "-EQ") token, because that
token is what the historical-data call needs.

No login is needed for this step.
Run:  python src/fno_universe.py
Output: data/meta/fno_universe.csv
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

SCRIP_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
META = Path(__file__).resolve().parents[1] / "data" / "meta"


def load_scrip_master(refresh: bool = False, max_age_hours: float = 12) -> pd.DataFrame:
    """Download (or reuse a recent copy of) the instrument master."""
    META.mkdir(parents=True, exist_ok=True)
    cache = META / "OpenAPIScripMaster.json"
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600
    if refresh or not fresh:
        print("Downloading instrument master (large file, one time per day) ...")
        r = requests.get(SCRIP_URL, timeout=180)
        r.raise_for_status()
        cache.write_bytes(r.content)
    with open(cache, encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def build_fno_universe(master: pd.DataFrame):
    """Return (universe DataFrame, list of F&O names we could not match to a cash token)."""
    m = master.copy()
    m["lotsize"] = pd.to_numeric(m["lotsize"], errors="coerce")

    fut = m[(m["exch_seg"] == "NFO") & (m["instrumenttype"] == "FUTSTK")].copy()
    fut["expiry_dt"] = pd.to_datetime(fut["expiry"], format="%d%b%Y", errors="coerce")
    fut = fut.sort_values("expiry_dt")
    fno = (fut.groupby("name")
              .agg(lotsize=("lotsize", "first"),
                   nearest_expiry=("expiry_dt", "first"),
                   n_contracts=("symbol", "count"))
              .reset_index())

    cash = (m[(m["exch_seg"] == "NSE") & (m["symbol"].str.endswith("-EQ"))]
              [["name", "symbol", "token"]]
              .drop_duplicates("name")
              .rename(columns={"symbol": "nse_symbol"}))

    out = fno.merge(cash, on="name", how="left")
    unmatched = out.loc[out["token"].isna(), "name"].tolist()
    out = (out.dropna(subset=["token"])
              .rename(columns={"name": "symbol"})
              .sort_values("symbol")
              .reset_index(drop=True))
    return out, unmatched


if __name__ == "__main__":
    master = load_scrip_master()
    universe, unmatched = build_fno_universe(master)
    path = META / "fno_universe.csv"
    universe.to_csv(path, index=False)
    print(f"F&O stocks found: {len(universe)}  ->  saved {path}")
    print(universe.head(10).to_string(index=False))
    if unmatched:
        print("\nF&O names with no cash-market match (check by hand):", unmatched)
