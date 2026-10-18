"""
STEP 5 - Check the downloaded data BEFORE trusting it.

Angel One daily prices are NOT adjusted for splits and bonus issues. A 1:1 bonus makes the
price halve overnight, which looks like a -50% crash and would wreck any 9- or 12-month
return calculation. This report lists suspicious days so we can fix them.

Run:  python src/check_data.py        ->  data/meta/data_quality.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "daily"
REVIEW, LIKELY = -0.25, -0.40            # one-day close-to-close moves


def check_frame(df: pd.DataFrame) -> dict:
    d = df.sort_values("Date").reset_index(drop=True)
    ret = d["Close"].pct_change()
    worst_i = ret.idxmin() if ret.notna().any() else None
    worst = float(ret.min()) if worst_i is not None else float("nan")
    gaps = d["Date"].diff().dt.days
    flag = "ok"
    if worst <= LIKELY:
        flag = "LIKELY_SPLIT_OR_BONUS"
    elif worst <= REVIEW:
        flag = "REVIEW"
    return dict(rows=len(d), first=d["Date"].min().date(), last=d["Date"].max().date(),
                worst_day=d.loc[worst_i, "Date"].date() if worst_i is not None else None,
                worst_move=round(worst, 3), biggest_gap_days=int(gaps.max()) if len(d) > 1 else 0,
                zero_volume_days=int((d["Volume"] == 0).sum()),
                bad_ohlc=int(((d["High"] < d[["Open", "Close"]].max(axis=1)) |
                              (d["Low"] > d[["Open", "Close"]].min(axis=1))).sum()),
                flag=flag)


def main():
    rows = []
    for f in sorted(RAW.glob("*.csv")):
        rep = check_frame(pd.read_csv(f, parse_dates=["Date"]))
        rep["symbol"] = f.stem
        rows.append(rep)
    rep = pd.DataFrame(rows).set_index("symbol").sort_values("worst_move")
    out = ROOT / "data" / "meta" / "data_quality.csv"
    rep.to_csv(out)
    print(rep.head(15).to_string())
    print("\nflags:", rep["flag"].value_counts().to_dict(), "->", out)


if __name__ == "__main__":
    main()
