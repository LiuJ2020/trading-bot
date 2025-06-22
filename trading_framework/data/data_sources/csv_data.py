import backtrader as bt
import backtrader.feeds as btfeeds
from backtrader.feeds import GenericCSVData, YahooFinanceCSVData
import pandas as pd
import os
from datetime import datetime
import yfinance as yf
from .config import STOCK_GROUPS, SEGMENTS
from typing import Union, List, Optional

def get_csv_data(
    segment_key: str,
    group_or_ticker: Union[str, List[str]],
    data_dir: Optional[str] = 'data/csv_data'
) -> Union[GenericCSVData, List[GenericCSVData]]:
    """
    For a group or ticker and segment, check if the CSV exists. If so, return it as a backtrader GenericCSVData.
    If a group is given, returns a list of GenericCSVData objects.
    If the file is not found, attempts to download it using download_segment_data.
    """
    segment = SEGMENTS[segment_key]
    segment_name = segment['name']

    # Determine tickers to load
    if isinstance(group_or_ticker, str):
        if group_or_ticker in STOCK_GROUPS:
            tickers = STOCK_GROUPS[group_or_ticker]
        else:
            tickers = [group_or_ticker]
    elif isinstance(group_or_ticker, list):
        tickers = group_or_ticker
    else:
        raise ValueError("group_or_ticker must be a group name, a ticker string, or a list of tickers.")

    datas = []
    missing = []
    for ticker in tickers:
        csv_path = os.path.join(data_dir, f"{ticker}_{segment_name}.csv")
        if os.path.exists(csv_path):
            data = GenericCSVData(
                dataname=csv_path,
                dtformat=('%Y-%m-%d'),
                datetime=0,
                open=1,
                high=2,
                low=3,
                close=4,
                volume=5,
                openinterest=-1
            )
            datas.append(data)
        else:
            missing.append(ticker)
    if missing:
        print(f"Missing files for: {missing}. Attempting to download...")
        download_segment_data(segment_key, missing, data_dir)
        # Try loading again after download
        for ticker in missing:
            csv_path = os.path.join(data_dir, f"{ticker}_{segment_name}.csv")
            if os.path.exists(csv_path):
                data = GenericCSVData(
                    dataname=csv_path,
                    dtformat=('%Y-%m-%d'),
                    datetime=0,
                    open=1,
                    high=2,
                    low=3,
                    close=4,
                    volume=5,
                    openinterest=-1
                )
                datas.append(data)
            else:
                print(f"Failed to download or find file for: {ticker}")
    if len(datas) == 1:
        return datas[0]
    return datas

# New function to batch download S&P 500 data for a segment
def download_segment_data(
    segment_key: str,
    group_or_ticker: Union[str, List[str]],
    data_dir: str = 'data/csv_data'
) -> None:
    """
    Downloads data for a given segment (date range) for a group (from STOCK_GROUPS) or a single ticker.
    Only downloads tickers that do not already have a CSV for this segment.
    """
    segment = SEGMENTS[segment_key]
    start = segment['start']
    end = segment['end']
    segment_name = segment['name']

    # Determine tickers to download
    if isinstance(group_or_ticker, str):
        if group_or_ticker in STOCK_GROUPS:
            tickers = STOCK_GROUPS[group_or_ticker]
        else:
            tickers = [group_or_ticker]
    elif isinstance(group_or_ticker, list):
        tickers = group_or_ticker
    else:
        raise ValueError("group_or_ticker must be a group name, a ticker string, or a list of tickers.")

    os.makedirs(data_dir, exist_ok=True)
    missing_tickers = []
    for ticker in tickers:
        csv_path = os.path.join(data_dir, f"{ticker}_{segment_name}.csv")
        if not os.path.exists(csv_path):
            missing_tickers.append(ticker)

    if not missing_tickers:
        print(f"All data for group/ticker {group_or_ticker} in segment {segment_name} already downloaded.")
        return

    print(f"Downloading {len(missing_tickers)} tickers for group/ticker {group_or_ticker} in segment {segment_name}...")
    # Download in batches of 50 to avoid API limits
    batch_size = 50
    for i in range(0, len(missing_tickers), batch_size):
        batch = missing_tickers[i:i+batch_size]
        df = yf.download(batch, start=start, end=end, group_by='ticker', auto_adjust=True, threads=True)
        for ticker in batch:
            csv_path = os.path.join(data_dir, f"{ticker}_{segment_name}.csv")
            # yfinance returns a multi-indexed DataFrame for multiple tickers
            if len(batch) == 1 and not isinstance(df.columns, pd.MultiIndex):
                tdf = df.copy()
                # tdf.index.name = 'datetime'
                tdf.to_csv(csv_path)
                print(f"Saved {csv_path}")
            elif ticker in df.columns.get_level_values(0):
                tdf = df[ticker].copy()
                # tdf.index.name = 'datetime'
                tdf.to_csv(csv_path)
                print(f"Saved {csv_path}")
            else:
                print(f"No data for {ticker} in segment {segment_name}")
