from typing import Optional
from .data_sources.csv_data import get_csv_data

def get_data_feed(
    segment_key: str,
    group_or_ticker: str,
    data_dir: Optional[str] = 'data/csv_data',
    mode: str = 'backtest'
):
    if mode == 'backtest':
        return get_csv_data(segment_key, group_or_ticker)
    elif mode == 'live':
        raise NotImplementedError("Live mode not yet supported")
