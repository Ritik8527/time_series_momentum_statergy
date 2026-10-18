"""Variation 1 - paper sizing (40% / volatility) versus fixed sizing (same size for every stock)."""
import matplotlib.pyplot as plt
import pandas as pd

from tsmom_core import buy_and_hold, run_tsmom, summarize
from tsmom_data import ROOT, clean_universe, load_close_prices

START, END = "2020-01-01", "2023-12-31"

prices, _ = clean_universe(load_close_prices())

runs = {}
for lb in (12, 9):
    runs[f"{lb}m paper sizing"] = run_tsmom(prices, lookback=lb, sizing="vol")
    runs[f"{lb}m fixed sizing"] = run_tsmom(prices, lookback=lb, sizing="fixed", fixed_size=1.0)

curves = {name: res["port"].loc[START:END] for name, res in runs.items()}
curves["Buy & hold (equal weight, 1x)"] = buy_and_hold(prices, runs["12m paper sizing"]).loc[START:END]

# ---- results table ----
table = pd.DataFrame({n: summarize(r) for n, r in curves.items()}).T
table["avg_|position|"] = [runs[n]["pos"].abs().mean(axis=1).loc[START:END].mean() if n in runs else 1.0
                           for n in table.index]
print(table.round(3).to_string())

yearly = pd.DataFrame({n: (1 + r).groupby(r.index.year).prod() - 1 for n, r in curves.items()})
print("\nreturn per calendar year:")
print(yearly.round(3).to_string())


# ---- which stocks drive the risk? share of total P&L variance from the 10 most volatile stocks ----
for name in ("12m paper sizing", "12m fixed sizing"):
    sr = runs[name]["stock_ret"].loc[START:END]
    spread = sr.std().sort_values(ascending=False)                    # monthly risk each stock added
    share = (spread.head(10) ** 2).sum() / (spread ** 2).sum()
    print(f"{name}: the 10 riskiest stocks produce {share:.0%} of the variance (equal share would be "
          f"{10 / len(spread):.0%})")


# ---- chart: colour = look-back, solid = paper sizing, dashed = fixed sizing ----
def to_equity(r):
    eq = (1 + r).cumprod()
    return pd.concat([pd.Series([1.0], index=[r.index[0] - pd.offsets.MonthEnd(1)]), eq])


style = {
    "12m paper sizing": ("#2a78d6", 2.2, "-"),
    "12m fixed sizing": ("#2a78d6", 1.8, "--"),
    "9m paper sizing": ("#eb6834", 2.2, "-"),
    "9m fixed sizing": ("#eb6834", 1.8, "--"),
    "Buy & hold (equal weight, 1x)": ("#52514e", 1.6, "-"),
}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1.3]})
for name, r in curves.items():
    eq = to_equity(r)
    color, lw, ls = style[name]
    ax1.plot(eq.index, eq, color=color, lw=lw, ls=ls, label=name)
    ax2.plot(eq.index, eq / eq.cummax() - 1, color=color, lw=lw, ls=ls)

ax1.set_title("TSMOM variation 1: paper sizing vs fixed sizing, 2020-2023 (before costs)", loc="left")
ax1.set_ylabel("Growth of 1.0")
ax2.set_ylabel("Drawdown")
ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax1.legend(frameon=False, loc="upper left")
for ax in (ax1, ax2):
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

out = ROOT / "results"
out.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(out / "tsmom_sizing_comparison.png", dpi=150)
print("\nsaved:", out / "tsmom_sizing_comparison.png")
plt.show()