"""TSMOM step 4 - equity curves, drawdowns and a results table for the in-sample window."""
import matplotlib.pyplot as plt
import pandas as pd

from tsmom_core import buy_and_hold, run_tsmom, summarize
from tsmom_data import ROOT, clean_universe, load_close_prices

START, END = "2020-01-01", "2023-12-31"

prices, _ = clean_universe(load_close_prices())
run12 = run_tsmom(prices, lookback=12)
run9 = run_tsmom(prices, lookback=9)
passive = run_tsmom(prices, lookback=12, passive_long=True)

curves = {                                                     # name -> monthly returns in the window
    "TSMOM 12m": run12["port"],
    "TSMOM 9m": run9["port"],
    "Buy & hold (equal weight, 1x)": buy_and_hold(prices, run12),
    "Passive long (same 40%/vol sizing)": passive["port"],
}
curves = {name: r.loc[START:END] for name, r in curves.items()}

# ---- table: headline numbers and calendar-year returns ----
print(pd.DataFrame({n: summarize(r) for n, r in curves.items()}).T.round(3).to_string())
yearly = pd.DataFrame({n: (1 + r).groupby(r.index.year).prod() - 1 for n, r in curves.items()})
print("\nreturn per calendar year:")
print(yearly.round(3).to_string())


# ---- equity curve and drawdown ----
def to_equity(r: pd.Series) -> pd.Series:
    """Monthly returns -> growth of 1.0, starting from 1.0 one month before the first return."""
    eq = (1 + r).cumprod()
    start = r.index[0] - pd.offsets.MonthEnd(1)
    return pd.concat([pd.Series([1.0], index=[start]), eq])


style = {                                                      # colour, line width, line style
    "TSMOM 12m": ("#2a78d6", 2.2, "-"),
    "TSMOM 9m": ("#eb6834", 2.2, "-"),
    "Buy & hold (equal weight, 1x)": ("#52514e", 1.8, "-"),
    "Passive long (same 40%/vol sizing)": ("#52514e", 1.4, "--"),
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1.3]})
for name, r in curves.items():
    eq = to_equity(r)
    color, lw, ls = style[name]
    ax1.plot(eq.index, eq, color=color, lw=lw, ls=ls, label=name)
    ax2.plot(eq.index, eq / eq.cummax() - 1, color=color, lw=lw, ls=ls)

ax1.set_title("TSMOM on Indian F&O stocks, raw rule, in-sample 2020-2023 (before costs)", loc="left")
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
fig.savefig(out / "tsmom_equity_2020_2023.png", dpi=150)
print("\nsaved:", out / "tsmom_equity_2020_2023.png")
plt.show()