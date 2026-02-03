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

def get_last_year_closing_prices_yf(ticker='BTC-USD'):
    df = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    last_date = df['Date'].max()
    start_date = last_date - pd.Timedelta(days=1825)
    df = df[df['Date'] >= start_date]
    closes = df['Adj_Close'].values
    dates = df['Date'].dt.strftime('%Y-%m-%d').values.tolist()
    return {'closes': closes, 'dates': dates}

def preprocess_for_prediction(closes, scaler, window_size):
    closes_scaled = scaler.transform(closes.reshape(-1, 1)).reshape(-1)
    return closes_scaled[-window_size:]

@app.route('/', methods=['GET', 'POST'])
def predict():
    predictions = None
    error = None
    window_size = 15
    hist_range = 60
    days = None
    future_dates = []
    coin = request.form.get('coin', 'BTC-USD')
    if coin != 'BTC-USD':
        error = "Prediction is only supported for Bitcoin at this time."
        latest_price = None
        last_prices = []
        last_dates = []
    else:
        data = get_last_year_closing_prices_yf('BTC-USD')
        if data is None or len(data['closes']) < window_size:
            error = "Price data is not available for this coin."
            latest_price = None
            last_prices = []
            last_dates = []
        else:
            closes = data['closes']
            dates = data['dates']
            
            latest_price = closes[-1]
            
            hist_range = int(request.form.get('hist_range', 60))
            
            last_prices = closes[-hist_range:]
            last_dates = dates[-hist_range:]
            
            if request.method == 'POST' and latest_price is not None:
                days = request.form.get('days')
                if days:
                    days = int(days)
                    model_input_closes = closes[-window_size:]
                    scaled_input = preprocess_for_prediction(model_input_closes, scaler, window_size).reshape(1, window_size, 1)
                    predictions = []
                    
                    last_date_str = dates[-1]
                    last_date_dt = datetime.strptime(last_date_str, '%Y-%m-%d')
                    future_dates = []
                    
                    for i in range(days):
                        scaled_pred = model.predict(scaled_input)[0][0]
                        pred = scaler.inverse_transform([[scaled_pred]])[0][0]
                        predictions.append(float(pred))
                        scaled_input = np.append(scaled_input[:, 1:, :], [[[scaled_pred]]], axis=1)
                        
                        next_date = last_date_dt + pd.Timedelta(days=i+1)
                        future_dates.append(next_date.strftime('%Y-%m-%d'))

    return render_template(
        'index.html',
        predictions=predictions,
        latest_price=latest_price if coin == 'BTC-USD' else None,
        last_prices=last_prices.tolist() if coin == 'BTC-USD' and last_prices is not None else [],
        last_dates=last_dates if coin == 'BTC-USD' and last_prices is not None else [],
        future_dates=future_dates if predictions else [],
        current_year=datetime.now().year,
        current_datetime=datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        coin=coin,
        error=error,
        days=days if days else 15,
        hist_range=hist_range if coin == 'BTC-USD' and last_prices is not None else 60
    )

@app.route('/about')
def about():
    return render_template('about.html', current_year=datetime.now().year)

if __name__ == '__main__':
    app.run(debug=True)