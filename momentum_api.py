import pandas as pd
import yfinance as yf
import numpy as np
import warnings
from flask import Flask, jsonify
from datetime import datetime, timedelta
import os
warnings.filterwarnings('ignore')


# Flask app initialization
app = Flask(__name__)

# Get today's date and subtract 4 years + 25 days
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=4 * 365 + 25)).strftime('%Y-%m-%d')

sp500_tickers = ['NVDA',
 'TSLA',
 'AAPL',
 'F',
 'AMZN',
 'AMD',
 'PLTR',
 'BAC',
 'T',
 'CCL',
 'INTC',
 'GOOGL',
 'PFE',
 'MSFT',
 'GOOG',
 'WFC',
 'UBER',
 'META',
 'AVGO',
 'WMT',
 'XOM',
 'SMCI',
 'KVUE',
 'VZ',
 'CSCO',
 'CMCSA',
 'C',
 'NCLH',
 'MU',
 'OXY',
 'WBD',
 'PCG',
 'FCX',
 'GM',
 'KMI',
 'KO',
 'PARA',
 'HBAN',
 'CMG',
 'LRCX',
 'PYPL',
 'DAL',
 'CSX',
 'UAL',
 'JPM',
 'KEY',
 'SLB',
 'DIS',
 'HPE',
 'BMY',
 'BA',
 'GE',
 'DVN',
 'HAL',
 'MRK',
 'CVX',
 'ORCL',
 'SCHW',
 'VTRS',
 'HPQ',
 'ANET',
 'NEE',
 'QCOM',
 'USB',
 'MS',
 'MO',
 'WBA',
 'RF',
 'MRNA',
 'NEM',
 'PANW',
 'JNJ',
 'BKR',
 'LUV',
 'BSX',
 'CTRA',
 'APA',
 'AMCR',
 'HST',
 'SBUX',
 'NKE',
 'WMB',
 'TFC',
 'CVS',
 'GILD',
 'COP',
 'V',
 'KDP',
 'EXC',
 'AMAT',
 'KHC',
 'PG',
 'ON',
 'CRM',
 'KR',
 'MDLZ',
 'EQT',
 'MGM',
 'EBAY',
 'NFLX',
 'RTX',
 'VICI',
 'ABBV',
 'LVS',
 'AES',
 'TJX',
 'ABNB',
 'FTNT',
 'MDT',
 'FITB',
 'TSCO',
 'ABT',
 'SYF',
 'CFG',
 'APH',
 'PPL',
 'DELL',
 'MOS',
 'MPC',
 'TXN',
 'MNST',
 'IBM',
 'PEP',
 'DOW',
 'TMUS',
 'GEN',
 'PM',
 'KIM',
 'RCL',
 'CARR',
 'BK',
 'CNP',
 'MCHP',
 'IVZ',
 'WDC',
 'VST',
 'GLW',
 'AIG',
 'MET',
 'CPRT',
 'CL',
 'JCI',
 'SO',
 'FIS',
 'O',
 'MMM',
 'CRWD',
 'DOC',
 'D',
 'TGT',
 'BX',
 'WY',
 'NI',
 'VLO',
 'EOG',
 'CAG',
 'FE',
 'MTCH',
 'IPG',
 'HD',
 'TPR',
 'GIS',
 'BAX',
 'JNPR',
 'DD',
 'ENPH',
 'INVH',
 'FI',
 'CNC',
 'CTVA',
 'FOXA',
 'CZR',
 'UPS',
 'AXP',
 'LOW',
 'CTSH',
 'IP',
 'ADI',
 'DXCM',
 'XEL',
 'PSX',
 'FAST',
 'BEN',
 'MA',
 'KKR',
 'UNH',
 'EW',
 'NRG',
 'PLD',
 'LLY',
 'ADM',
 'CAT',
 'OKE',
 'DUK',
 'DHI',
 'ADBE',
 'AEP',
 'WYNN',
 'GEHC',
 'EMR',
 'HWM',
 'MCD',
 'HON',
 'SRE',
 'DHR',
 'COF',
 'AFL',
 'UNP',
 'NWSA',
 'BBY',
 'PCAR',
 'ETR',
 'EXPE',
 'AMGN',
 'PGR',
 'SYY',
 'K',
 'CF',
 'GS',
 'ICE',
 'STX',
 'APO',
 'PHM',
 'NDAQ',
 'DLTR',
 'CPB',
 'TSN',
 'IR',
 'LEN',
 'BWA',
 'FANG',
 'PEG',
 'WELL',
 'NXPI',
 'CAH',
 'FDX',
 'STT',
 'SPG',
 'ROST',
 'DECK',
 'CSGP',
 'DG',
 'VTR',
 'EA',
 'NUE',
 'CEG',
 'FSLR',
 'HES',
 'CCI',
 'ACN',
 'FTV',
 'OTIS',
 'PNC',
 'HRL',
 'LYV',
 'COST',
 'UDR',
 'DFS',
 'BLDR',
 'MAR',
 'WDAY',
 'HLT',
 'GPN',
 'ED',
 'ACGL',
 'ZTS',
 'LYB',
 'CMS',
 'APTV',
 'AMT',
 'EQR',
 'HIG',
 'MAS',
 'SWKS',
 'EIX',
 'BALL',
 'IRM',
 'PRU',
 'ETN',
 'TRGP',
 'NTAP',
 'KMB',
 'EL',
 'IFF',
 'WRB',
 'LIN',
 'DLR',
 'ES',
 'STLD',
 'TAP',
 'ALL',
 'CB',
 'HOLX',
 'PAYX',
 'ALB',
 'CI',
 'ISRG',
 'MMC',
 'TER',
 'ADP']

df = yf.download(sp500_tickers, start=start_date, end=end_date)
sp500 = df['Adj Close'].dropna(how='all', axis=1)

time_period = 1008  # 4 years ~ 252 days/year
lag = 20            # ~1 month

# ------------------------
#   Momentum Calculation
# ------------------------

def calculate_momentum_factors(how_many_days_back=0):
    start_time = how_many_days_back + time_period + lag
    most_current_time = how_many_days_back + lag

    lagged_closed_price = sp500[-start_time : -most_current_time]
    rolling_mean_price = lagged_closed_price.rolling(window=time_period - 251).mean().dropna(how='all')

    # 52-week trend
    slope_info = pd.DataFrame(index=sp500.columns)
    for i in range(1, lag + 1):
        slope_info[i] = rolling_mean_price.apply(
            lambda x: np.polyfit(
                np.arange(len(x[-i - 252 : -i])),
                x[-i - 252 : -i], 1
            )[0] if x[-i - 252 : -i].notna().all() else np.nan
        )
    _52_week_trend = slope_info.mean(axis=1)

    # % above 260-day low
    percent_above_260 = ((lagged_closed_price 
                          - lagged_closed_price.rolling(window=260).min())
                         / lagged_closed_price.rolling(window=260).min()
                         * 100).dropna(how='all').mean()

    # 4/52-week oscillator
    oscillator = ((lagged_closed_price.rolling(window=4*5).mean()
                   - lagged_closed_price.rolling(window=52*5).mean())
                  / lagged_closed_price.rolling(window=52*5).mean()
                  * 100).dropna(how='all').mean()

    # 39-week return
    returns_39w = lagged_closed_price.pct_change(periods=39 * 5).dropna(how='all').mean()

    # 51-week Volume Price Trend
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
    z_scores = (x - x.mean()) / x.std()
    z_scores = z_scores.sum(axis=1)
    return z_scores.sort_values(ascending=False)

def get_long_basket():
    momentum_factors = calculate_momentum_factors(0)
    long_basket = calculate_z_scores(momentum_factors)[:10]
    return long_basket


# ------------------------
#   Flask API Routes
# ------------------------

@app.route('/api/momentum', methods=['GET'])
def momentum():
    long_basket = get_long_basket()
    short_basket = get_short_basket()

    long_list = [{"ticker": idx, "score": float(val)} for idx, val in long_basket.items()]

    return jsonify({
        "longBasket": long_list,
    })

@app.route('/')
def index():
    return "Momentum API is running. Visit /api/momentum to fetch data."

if __name__ == '__main__':
    from waitress import serve
    port = int(os.environ.get("PORT", 8080))  # Default to 8080 if Railway doesn't set it
    print(f"Starting server on port {port}...")
    serve(app, host='0.0.0.0', port=port)