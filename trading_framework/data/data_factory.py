from data.data_sources.csv_data import get_csv_data

def get_data_feed(mode='backtest'):
    if mode == 'backtest':
        return get_csv_data("data/sample.csv")
    elif mode == 'live':
        raise NotImplementedError("Live mode not yet supported")
