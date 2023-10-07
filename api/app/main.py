from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.responses import HTMLResponse
from joblib import load
import pandas as pd
import numpy as np
from functools import lru_cache
import gc

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def read_root():
    info = '''
        <h1>Welcome!</h1>
        <p>This web app is made for Predicting and Forecasting the sales revenue in 10 retail stores across 3 different states in the US.</p>
        <ol>
        <li><strong>The prediction model predicts the approximate revenue for a given item with its average sell price, store id, and a given date.</strong></li> 
        <li><strong>The forcasting model gives the forecasted sales revenue for the next 7 days.</strong></li>
        <h2>The following are the accessible API endpoints:</h2>
        <ol>
        <li>/health/ - Status code 200 with a welcome message.</li>
        <li>/sales/national/ - Returns next 7 days sales revenue forecast.</li>
        <li>/sales/stores/items/ - Returns predicted sales revenue for an input item, sell price, store, and date.</li>
        </ol>

        <h2>Expected input parameters for /sales/stores/items/:</h2>
        <ul>
        <li>item_id: string</li>
        <li>store_id: string</li>
        <li>sell_price: float</li>
        <li>date: string, MUST be in mm/dd/yyyy format</li>
        </ul>

        <h2>Model output: [List]</h2>

        <p><strong>Github link:</strong> <a href="https://github.com/Kritz23/adv_mla_at2">https://github.com/Kritz23/adv_mla_at2</a></p>
        '''
    gc.collect()
    return HTMLResponse(content=info)

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
@lru_cache(maxsize=None) 
def forecast():
    # Load the model if it's not in the cache
    if forecast.cache_info().misses == 1:
        arima_pipe = load('../models/arima.joblib')

    forecasted_sales = arima_pipe.predict(n_periods=7)
    gc.collect()
    return JSONResponse(np.round(forecasted_sales, decimals=2).tolist())

### sales revenue prediction
@app.get("/sales/stores/items")
@lru_cache(maxsize=None) 
def predict(
    item_id: str,
    store_id: str,
    sell_price: float,
    date: str,
):
    # Load the model if it's not in the cache
    if predict.cache_info().misses == 1:
        sgd_pipe = load('../models/sgd_pipeline.joblib')

    features = format_features(
        item_id,
        store_id,
        sell_price,
        date
        )
    
    obs = pd.DataFrame(features)
    print(obs, flush=True)
    pred = sgd_pipe.predict(obs)
    gc.collect()
    return JSONResponse(pred.tolist())
