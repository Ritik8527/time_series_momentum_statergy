"""TSMOM step 2 - the strategy itself: signal, volatility, position size, monthly return."""
import numpy as np
import pandas as pd

TRADING_DAYS = 261          # the paper's number for annualising


def month_end_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily close table -> one row per month holding the last available close of that month."""
    return prices.resample("ME").last()


def annual_vol(prices: pd.DataFrame, com: float = 60) -> pd.DataFrame:
    """Ex-ante annualised volatility of each stock, from exponentially weighted daily returns."""
    daily_ret = prices.pct_change(fill_method=None)
    daily_vol = daily_ret.ewm(com=com, min_periods=60).std()           # recent days weigh more
    return daily_vol * np.sqrt(TRADING_DAYS)


def run_tsmom(prices: pd.DataFrame, lookback: int = 12, target_vol: float = 0.40,
              cap: float | None = None, passive_long: bool = False,
              sizing: str = "vol", fixed_size: float = 1.0) -> dict:
    """Run the monthly TSMOM rule.

    lookback     months of past return used for the signal (12 = paper, also try 9)
    target_vol   risk per position, 0.40 = 40% annualised (used when sizing="vol")
    cap          optional maximum leverage per stock (None = no cap, like the paper)
    passive_long if True, ignore the signal and always go long (comparison line)
    sizing       "vol"   = the paper: position = target_vol / stock volatility
                 "fixed" = every stock gets the same size, whatever its volatility
    fixed_size   size used when sizing="fixed" (1.0 = 1x the stock's value)
    """
    me = month_end_prices(prices)                                      # month-end prices
    past = me / me.shift(lookback) - 1                                 # return over the look-back window
    signal = np.sign(past)                                             # +1 / -1 / 0, NaN if no history
    if passive_long:
        signal = signal.where(signal.isna(), 1.0)                      # keep NaN, make everything else +1

    vol_m = annual_vol(prices).resample("ME").last()                   # vol known at each month end
    if sizing == "vol":
        size = target_vol / vol_m                                      # position size = 40% / sigma
        if cap is not None:
            size = size.clip(upper=cap)
    elif sizing == "fixed":
        size = pd.DataFrame(fixed_size, index=vol_m.index, columns=vol_m.columns)
        size = size.where(vol_m.notna())                               # same stock-months as the paper version
    else:
        raise ValueError("sizing must be 'vol' or 'fixed'")
    pos = signal * size                                                # signed position set at month end

    ret_m = me.pct_change(fill_method=None)                            # each month's stock return
    stock_ret = pos.shift(1) * ret_m                                   # position from LAST month end x THIS month's return
    port = stock_ret.mean(axis=1, skipna=True)                         # equal weight across stocks that have a value
    n_active = stock_ret.notna().sum(axis=1)
    port = port[n_active > 0]
    return {"port": port, "stock_ret": stock_ret, "pos": pos, "signal": signal, "n_active": n_active}


def summarize(r: pd.Series) -> dict:
    """Headline numbers for a series of monthly returns."""
    r = r.dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    equity = (1 + r).cumprod()
    return {
        "months": len(r),
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": r.mean() * 12 / ann_vol,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "hit_rate": (r > 0).mean(),
    }


def buy_and_hold(prices: pd.DataFrame, like: dict) -> pd.Series:
    """Equal-weight buy & hold, 1x, no leverage: average monthly return of the SAME stocks
    and months that the strategy `like` (a run_tsmom result) traded."""
    ret_m = month_end_prices(prices).pct_change(fill_method=None)
    ret_m = ret_m.where(like["stock_ret"].notna())          # keep only stock-months the strategy had
    return ret_m.mean(axis=1).dropna()