import pandas as pd
import yfinance as yf
import numpy as np
import warnings
from flask import Flask, jsonify
from datetime import datetime, timedelta
import os
from waitress import serve

warnings.filterwarnings('ignore')

# Flask app initialization
app = Flask(__name__)

# Get today's date and subtract 4 years + 25 days
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=4 * 365 + 25)).strftime('%Y-%m-%d')

# Top 300 S&P 500 stocks
sp500_tickers = [
    'NVDA', 'TSLA', 'AAPL', 'F', 'AMZN', 'AMD', 'PLTR', 'BAC', 'T', 'CCL',
    'INTC', 'GOOGL', 'PFE', 'MSFT', 'GOOG', 'WFC', 'UBER', 'META', 'AVGO', 'WMT',
    'XOM', 'SMCI', 'KVUE', 'VZ', 'CSCO', 'CMCSA', 'C', 'NCLH', 'MU', 'OXY'
]

# Retrieve stock data
df = yf.download(sp500_tickers, start=start_date, end=end_date)
sp500 = df['Adj Close'].dropna(how='all', axis=1)

# Define momentum calculation parameters
time_period = 1008  # 4 years (252 days/year)
lag = 20  # 1 month

def calculate_momentum_factors(how_many_days_back=0):
    start_time = how_many_days_back + time_period + lag
    most_current_time = how_many_days_back + lag

    lagged_closed_price = sp500[-start_time: -most_current_time]
    rolling_mean_price = lagged_closed_price.rolling(window=time_period - 251).mean().dropna(how='all')

    # 52-week trend
    slope_info = pd.DataFrame(index=sp500.columns)
    for i in range(1, lag + 1):
        slope_info[i] = rolling_mean_price.apply(
            lambda x: np.polyfit(
                np.arange(len(x[-i - 252: -i])),
                x[-i - 252: -i], 1
            )[0] if x[-i - 252: -i].notna().all() else np.nan
        )
    _52_week_trend = slope_info.mean(axis=1)

    # % above 260-day low
    percent_above_260 = ((lagged_closed_price - lagged_closed_price.rolling(window=260).min())
                         / lagged_closed_price.rolling(window=260).min() * 100).dropna(how='all').mean()

    # 4/52-week oscillator
    oscillator = ((lagged_closed_price.rolling(window=4*5).mean()
                   - lagged_closed_price.rolling(window=52*5).mean())
                  / lagged_closed_price.rolling(window=52*5).mean() * 100).dropna(how='all').mean()

    # 39-week return
    returns_39w = lagged_closed_price.pct_change(periods=39 * 5).dropna(how='all').mean()

    # Creating DataFrame
    new_table = pd.DataFrame(index=sp500.columns)
    new_table['Slope 52 Week Trend-Line'] = _52_week_trend
    new_table['Percent above 260 Day Low'] = percent_above_260
    new_table['4/52 Week Oscillator'] = oscillator
    new_table['39_Week_Return'] = returns_39w

    return new_table

def calculate_z_scores(x):
    z_scores = (x - x.mean()) / x.std()
    z_scores = z_scores.sum(axis=1)
    return z_scores.sort_values(ascending=False)

def get_long_basket():
    momentum_factors = calculate_momentum_factors(0)
    long_basket = calculate_z_scores(momentum_factors)[:10]
    return long_basket

# API Routes
@app.route('/api/momentum', methods=['GET'])
def momentum():
    long_basket = get_long_basket()
    long_list = [{"ticker": idx, "score": float(val)} for idx, val in long_basket.items()]
    
    return jsonify({"longBasket": long_list})

@app.route('/')
def index():
    return "Momentum API is running. Visit /api/momentum to fetch data."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    serve(app, host='0.0.0.0', port=port)
