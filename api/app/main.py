from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

sgd_pipe = load('../models/sgd_pipeline.joblib')
arima_model = load('../models/arima.joblib')

@app.get("/")
def read_root():
    info = '''
            Welcome! This web app is made for Predicting and Forecasting \
            the sales revenue in 10 retail stores accross 3 diff states in the US. \n 
            1. The prediction model predicts the approximate revenue for a given item with \
            its average sell price, store id, and a given date. \n 
            2. The forcasting model gives the forecasted sales revenue for the \
            next 7 days. \n The following are the accessible API endpoints" \n 
            1. /health/ - Status code 200 with a welcome message. \n 
            2. /sales/national/ - Returns next 7 days sales revenue forecast \n 
            3. /sales/stores/items/ - Returns predicted sales revenue for an input item, sell price, store and date. \n
            Expected input parameters for /sales/stores/items/: \n 
            item_id: string,
            store_id: string,
            sell_price: float,
            date: string \n
            Model output: list \n \n 
            Github link = https://github.com/Kritz23/adv_mla_at2
            '''
    return info

@app.get('/health', status_code=200)
def healthcheck():
    return 'Predictive and Forecasting models are ready to use!'

def format_features(
    item_id: str,
    store_id: str,
    sell_price: float,
    date: str,
    ):
    return {
        'item_id': [item_id],
        'store_id': [store_id],
        'sell_price': [sell_price],
        'year': [pd.to_datetime(date).year],
        'month': [pd.to_datetime(date).month],
        'day': [pd.to_datetime(date).day],
        'weekday': [pd.to_datetime(date).dayofweek]
    }

### 7 days sales forecast
@app.get("/sales/national")
def predict():
    forecast = arima_model.predict(n_periods=7)
    return JSONResponse(forecast.tolist())

### sales revenue prediction
@app.get("/sales/stores/items")
def predict(
    item_id: str,
    store_id: str,
    sell_price: float,
    date: str,
):
    features = format_features(
        item_id,
        store_id,
        sell_price,
        date
        )
    obs = pd.DataFrame(features)
    print(obs, flush=True)
    pred = sgd_pipe.predict(obs)
    return JSONResponse(pred.tolist())
