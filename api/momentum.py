import json
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image
import os
import sys
import plotly
import plotly.graph_objs as go

# ---------------------------
#   YOUR MOMENTUM CODE
# ---------------------------

# The SP500 is the benchmark, its updated 500 stocks are sourced from the official daily-updated csv from the State Street Global Advisors website.
sp500_tickers = pd.read_excel(
    "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx",
    header=4
).Ticker.dropna().to_list()

# Retrieve historical data
df = yf.download(sp500_tickers, period='10y')
sp500 = df['Adj Close'].dropna(how='all', axis=1)

time_period = 1260  # 5 years of trading days
lag = 20  # about 1 month of trading days

def calculate_momentum_factors(how_many_days_back):
    start_time = how_many_days_back + time_period + lag
    most_current_time = how_many_days_back + lag

    lagged_closed_price = sp500[-start_time : -most_current_time]
    rolling_mean_price = lagged_closed_price.rolling(window=time_period - 251).mean().dropna(how='all')

    # Calculate the 52-week trend
    slope_info = pd.DataFrame(index=sp500.columns)
    for i in range(1, lag + 1):
        slope_info[i] = rolling_mean_price.apply(
            lambda x: np.polyfit(
                np.arange(len(x[-i - 252 : -i])),
                x[-i - 252 : -i],
                1
            )[0] if x[-i - 252 : -i].notna().all() else np.nan
        )
    _52_week_trend = slope_info.mean(axis=1)

    # Percent above 260 day low
    percent_above_260 = (
        (lagged_closed_price - lagged_closed_price.rolling(window=260).min())
        / lagged_closed_price.rolling(window=260).min() * 100
    ).dropna(how='all').mean()

    # 4/52 Week Oscillator
    oscillator = (
        (lagged_closed_price.rolling(window=4*5).mean()
         - lagged_closed_price.rolling(window=52*5).mean())
        / lagged_closed_price.rolling(window=52*5).mean() * 100
    ).dropna(how='all').mean()

    # 39-Week Return
    returns_39w = lagged_closed_price.pct_change(periods=39 * 5).dropna(how='all').mean()

    # 51 Week Volume Price Trend
    volume = df['Volume'].dropna(how='all', axis=1)[-lag - time_period : -lag]
    vpt = (volume * lagged_closed_price.pct_change()).cumsum()
    vpt_51w = (vpt - vpt.shift(periods=51 * 5)).dropna(how='all').mean()

    new_table = pd.DataFrame(index=sp500.columns)
    new_table['Slope 52 Week Trend-Line'] = _52_week_trend
    new_table['Percent above 260 Day Low'] = percent_above_260
    new_table['4/52 Week Oscillator'] = oscillator
    new_table['39_Week_Return'] = returns_39w
    new_table['51 Week Volume Price Trend'] = vpt_51w
    return new_table

def calculate_z_scores(x):
    # Standard z-score formula across all columns, then sum them for a total momentum measure
    z_scores = (x - x.mean()) / x.std()
    z_scores = z_scores.sum(axis=1)
    return z_scores.sort_values(ascending=False)

def get_long_basket():
    momentum_factors = calculate_momentum_factors(0)
    return calculate_z_scores(momentum_factors)[:10]

def get_short_basket():
    momentum_factors = calculate_momentum_factors(0)
    return calculate_z_scores(momentum_factors)[-10:]

# -------------
#   VERCEL HANDLER
# -------------
def handler(request):
    """
    Called by Vercel's Python serverless environment.
    We'll return JSON with both the long and short baskets.
    """
    long_basket = get_long_basket()  # pd.Series
    short_basket = get_short_basket()  # pd.Series

    long_list = [
        {"ticker": str(idx), "score": float(val)}
        for idx, val in long_basket.items()
    ]
    short_list = [
        {"ticker": str(idx), "score": float(val)}
        for idx, val in short_basket.items()
    ]

    # We must return a dict with 'statusCode', 'headers', 'body' for serverless
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "longBasket": long_list,
            "shortBasket": short_list
        })
    }
