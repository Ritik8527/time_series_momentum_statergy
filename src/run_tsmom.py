"""TSMOM step 3 - run the raw rule on the cleaned in-sample data and print the results."""
import pandas as pd

from tsmom_core import run_tsmom, summarize
from tsmom_data import clean_universe, load_close_prices

START, END = "2020-01-01", "2023-12-31"        # the in-sample window we REPORT on (2018-19 is warm-up only)

prices, _ = clean_universe(load_close_prices())
print("stocks used:", prices.shape[1])

rows = {}
for lb in (12, 9):                                                   # look-back in months
    res = run_tsmom(prices, lookback=lb)
    r = res["port"].loc[START:END]                                   # keep only the reporting window
    rows[f"TSMOM {lb}m"] = summarize(r)
    gross = res["pos"].abs().mean(axis=1).loc[START:END].mean()      # average |position| per stock
    print(f"{lb}m: average position size per stock = {gross:.2f}x, "
          f"stocks active per month = {res['n_active'].loc[START:END].mean():.0f}")
    print((1 + r).groupby(r.index.year).prod() - 1, "\n")            # return per calendar year

passive = run_tsmom(prices, lookback=12, passive_long=True)["port"].loc[START:END]
rows["Passive long (same sizing)"] = summarize(passive)

print(pd.DataFrame(rows).T.round(3).to_string())