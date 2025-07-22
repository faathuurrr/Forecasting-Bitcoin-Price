import requests
import numpy as np
import pickle
import pandas as pd
import yfinance as yf
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from datetime import datetime

app = Flask(__name__)

model = load_model('../model/model.h5')
with open('../model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# def get_last_n_closing_prices(coin='bitcoin', n=15):
#     url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={n}&interval=daily"
    
# #    headers = {
# #         "x-cg-pro-api-key": "CG-cDtXdGQV4XwYHqcZXHKG9gDS"
# #     }
# #     response = requests.get(url, headers=headers)
#     response = requests.get(url)
#     data = response.json()
    
#     if 'prices' not in data:
#         return None
#     closes = [price[1] for price in data['prices']]
#     return np.array(closes[-n:])

def get_last_year_closing_prices_yf(ticker='BTC-USD'):
    df = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    last_date = df['Date'].max()
    start_date = last_date - pd.Timedelta(days=365)
    df = df[df['Date'] >= start_date]
    closes = df['Adj_Close'].values
    return closes

def preprocess_for_prediction(closes, scaler, window_size):
    closes_scaled = scaler.transform(closes.reshape(-1, 1)).reshape(-1)
    return closes_scaled[-window_size:]

@app.route('/', methods=['GET', 'POST'])
def predict():
    predictions = None
    error = None
    window_size = 15
    coin = request.form.get('coin', 'BTC-USD')
    if coin != 'BTC-USD':
        error = "Prediction is only supported for Bitcoin at this time."
        latest_price = None
        last_prices = []
    else:
        closes = get_last_year_closing_prices_yf('BTC-USD')
        if closes is None or len(closes) < window_size:
            error = "Price data is not available for this coin."
            latest_price = None
            last_prices = []
        else:
            latest_price = closes[-1]
            last_prices = closes[-window_size:]
            if request.method == 'POST' and latest_price is not None:
                days = request.form.get('days')
                if days:
                    days = int(days)
                    scaled_input = preprocess_for_prediction(closes, scaler, window_size).reshape(1, window_size, 1)
                    predictions = []
                    for _ in range(days):
                        scaled_pred = model.predict(scaled_input)[0][0]
                        pred = scaler.inverse_transform([[scaled_pred]])[0][0]
                        predictions.append(float(pred))
                        scaled_input = np.append(scaled_input[:, 1:, :], [[[scaled_pred]]], axis=1)

    return render_template(
        'index.html',
        predictions=predictions,
        latest_price=latest_price if coin == 'BTC-USD' else None,
        last_prices=last_prices.tolist() if coin == 'BTC-USD' and last_prices is not None else [],
        current_year=datetime.now().year,
        coin=coin,
        error=error
    )

@app.route('/about')
def about():
    return render_template('about.html', current_year=datetime.now().year)

if __name__ == '__main__':
    app.run(debug=True)