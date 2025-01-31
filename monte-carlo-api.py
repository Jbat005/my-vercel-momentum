import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from waitress import serve
from datetime import datetime, timedelta
import os

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Define start and end dates for fetching stock data
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=4 * 365 + 25)).strftime('%Y-%m-%d')

def fetch_stock_data(tickers):
    """Fetch historical stock data for selected tickers."""
    try:
        print(f"Fetching stock data for: {tickers}")  # Debugging Step
        df = yf.download(tickers, start=start_date, end=end_date, progress=False)

        if df.empty:
            raise ValueError("No data retrieved for the given tickers.")

        return df['Adj Close'].dropna(how='all', axis=1)

    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def monte_carlo_simulation(tickers, num_portfolios=50000):
    """Run Monte Carlo simulation on selected stocks."""
    try:
        sp500 = fetch_stock_data(tickers)
        
        # Handle empty stock data scenario
        if sp500 is None or sp500.empty:
            print("Error: No stock data retrieved.")
            return {"error": "No stock data available for the given tickers"}

        print(f"Stock data retrieved: {sp500.shape}")  # Debugging Step

        returns = sp500.pct_change().dropna()

        # Ensure returns DataFrame has valid data
        if returns.empty:
            print("Error: No valid returns data available.")
            return {"error": "No valid returns data"}

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

            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = sharpe_ratio
            weights_record.append(weights)

        # Identify the max Sharpe Ratio portfolio
        max_sharpe_idx = np.argmax(results[2])
        max_sharpe_weights = weights_record[max_sharpe_idx]

        # Identify the minimum volatility portfolio
        min_vol_idx = np.argmin(results[1])
        min_vol_weights = weights_record[min_vol_idx]

        return {
            "MinVolPortfolio": {
                "weights": dict(zip(tickers, min_vol_weights)),
                "return": results[0, min_vol_idx],
                "volatility": results[1, min_vol_idx],
                "sharpe": results[2, min_vol_idx],
            },
            "MaxSharpePortfolio": {
                "weights": dict(zip(tickers, max_sharpe_weights)),
                "return": results[0, max_sharpe_idx],
                "volatility": results[1, max_sharpe_idx],
                "sharpe": results[2, max_sharpe_idx],
            },
            "numPortfolios": num_portfolios
        }

    except Exception as e:
        print(f"Monte Carlo Simulation Error: {str(e)}")  # Debugging Step
        return {"error": f"Internal error: {str(e)}"}

@app.route('/api/run-monte-carlo', methods=['POST'])
def run_monte_carlo():
    """Endpoint for running Monte Carlo simulation."""
    try:
        data = request.get_json()
        tickers = data.get("tickers", [])
        num_portfolios = data.get("num_portfolios", 50000)

        if not tickers or len(tickers) < 2:
            return jsonify({"error": "Please provide at least two stocks."}), 400

        mc_result = monte_carlo_simulation(tickers, num_portfolios)

        if mc_result:
            return jsonify(mc_result)
        else:
            return jsonify({"error": "Failed to run simulation"}), 500
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/')
def index():
    return "Monte Carlo Simulation API is running."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    serve(app, host='0.0.0.0', port=port)
