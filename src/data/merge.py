import pandas as pd

cal = pd.read_csv("../../data/raw/calendar.csv")
items_price = pd.read_csv("../../data/raw/items_weekly_sell_prices.csv")

class Merger:
    """ 
    Merge date and item sell price, calculate items's daily sales revenue
    """
    def __init__(self):
        self.cal = cal
        self.items_price = items_price

    def merge_df(self, df):

        # change the data type of categorical columns to pandas category type: saves memory
        df['item_id'] = df['item_id'].astype("category")
        df['store_id'] = df['store_id'].astype("category")

        # merge calender and item price
        df_cal = df.merge(self.cal[['d', 'date', 'wm_yr_wk']], on='d', how='left')
        df_cal['date'] = pd.to_datetime(df_cal['date'])
        df_cal.drop(["d"], inplace=True, axis=1)
        df_merged = df_cal.merge(self.items_price, on=['wm_yr_wk', 'item_id', 'store_id'], how='inner')

        # downcast to numeric columns to float 32: uses less memory
        df_merged[["units_sold", "wm_yr_wk"]] = df_merged[["units_sold", "wm_yr_wk"]].apply(pd.to_numeric, downcast="float")
        # calculate item's daily sales revenue
        df_merged["revenue"] = df_merged["units_sold"]*df_merged["sell_price"]
        df_merged.drop(["wm_yr_wk", "units_sold"], inplace=True, axis=1)

        return df_merged