from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

sgd_pipe = load('../models/sgd_pipeline.joblib')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'SGDRegressor is all ready to go!'

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

# Solution:
@app.get("/sales/stores/items/prediction")
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
