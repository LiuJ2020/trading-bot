import backtrader as bt
import pandas as pd

def get_csv_data(filepath):
    df = pd.read_csv(filepath, index_col='datetime', parse_dates=True)
    data = bt.feeds.PandasData(dataname=df)
    return data
