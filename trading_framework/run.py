import argparse
from engine.backtest_runner import run_backtest
from strategies.moving_average_crossover import MovingAverageStrategy

def main():
    parser = argparse.ArgumentParser(description="Run a backtest for a given segment and group/ticker.")
    parser.add_argument("segment_key", type=str, help="Segment key (e.g., covid, bull, precrisis, qe, expansion)")
    parser.add_argument("group_or_ticker", type=str, help="Group name (e.g., SP500, ETFs) or individual ticker (e.g., AAPL)")
    parser.add_argument("--data_dir", type=str, default='data/csv_data', help="Directory for CSV data files")
    args = parser.parse_args()
    run_backtest(MovingAverageStrategy, args.segment_key, args.group_or_ticker, data_dir=args.data_dir)

if __name__ == "__main__":
    main()