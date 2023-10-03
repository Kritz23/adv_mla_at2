import pandas as pd 
import sys
sys.path.append('../../src/data')
from get_date import get_date_features

def predict_df(df_new, pipeline):
    df_new = get_date_features(df_new)
    predicted_sales = pipeline.predict(df_new)

    return predicted_sales