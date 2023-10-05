import pandas as pd

def get_date_features(df_merged):
    df = df_merged.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.dayofweek
    
    return df.drop(["date"], axis=1)