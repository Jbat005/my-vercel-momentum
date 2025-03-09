import pandas as pd
import yfinance as yf
import numpy as np
import warnings
from datetime import datetime, timedelta
import os
from waitress import serve
from flask_cors import CORS  
from flask import Flask, jsonify, request, make_response

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

warnings.filterwarnings('ignore')

# Get date ranges
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
sp500 = df['Close'].dropna(how='all', axis=1)

# Momentum calculation parameters
time_period = 1008  # 4 years (252 days/year)
lag = 20  # 1 month

def calculate_momentum_factors(how_many_days_back):
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

    # Create DataFrame
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

def monte_carlo_simulation(tickers, num_portfolios):
    """Run Monte Carlo simulation on long basket"""
    try:
        prices = sp500[tickers]
        returns = prices.pct_change().dropna()
        
        # Portfolio metrics calculation
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)  # Normalize to sum to 1.0
            
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
            sharpe_ratio = portfolio_return / portfolio_volatility
            
            results[0,i] = portfolio_return
            results[1,i] = portfolio_volatility 
            results[2,i] = sharpe_ratio
            weights_record.append(weights)
            
        max_sharpe_idx = np.argmax(results[2])
        optimal_weights = weights_record[max_sharpe_idx]
        
        return {
            'weights': optimal_weights,
            'return': results[0,max_sharpe_idx],
            'volatility': results[1,max_sharpe_idx],
            'sharpe': results[2,max_sharpe_idx]
        }
        
    except Exception as e:
        print(f"Monte Carlo error: {str(e)}")
        return None

@app.route('/api/momentum', methods=['GET'])
def momentum():
    try:

        mc_result = monte_carlo_simulation(get_long_basket().index.tolist(), 100)
        
        if mc_result:
            optimized_weights = [
                {"ticker": t, "weight": float(w)} 
                for t, w in zip(get_long_basket().index.tolist(), mc_result['weights'])
            ]
            metrics = {
                "expectedReturn": float(mc_result['return']),
                "volatility": float(mc_result['volatility']),
                "sharpeRatio": float(mc_result['sharpe'])
            }
        else:
            optimized_weights = []
            metrics = {
                "expectedReturn": 0,
                "volatility": 0,
                "sharpeRatio": 0
            }
        
        response = jsonify({
            "optimizedWeights": optimized_weights,
            "portfolioMetrics": metrics
        })
        
    except Exception as e:
        print(f"API error: {str(e)}")
        response = jsonify({"error": "Internal server error"})
        response.status_code = 500

        
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/api/momentum', methods=['OPTIONS'])
def options():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    return response

@app.route('/')
def index():
    return "Momentum API with Monte Carlo Optimization"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    serve(app, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))