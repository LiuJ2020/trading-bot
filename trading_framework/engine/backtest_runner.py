import backtrader as bt
from data.data_factory import get_data_feed
from broker.broker_factory import get_broker
from analyzers.analyzer_suite import attach_analyzers
from sizers.percent_risk_sizer import PercentageRiskSizer

def run_backtest(strategy_class: type, segment_key: str, group_or_ticker: str, data_dir: str = 'data/csv_data') -> None:
    cerebro = bt.Cerebro()

    # Data
    data_feed = get_data_feed(segment_key=segment_key, group_or_ticker=group_or_ticker, data_dir=data_dir, mode='backtest')
    if isinstance(data_feed, list):
        for feed in data_feed:
            cerebro.adddata(feed)
    else:
        cerebro.adddata(data_feed)

    # Sizer (default: PercentRiskSizer)
    cerebro.addsizer(PercentageRiskSizer)

    # Strategy
    cerebro.addstrategy(strategy_class)

    # Broker setup
    broker = get_broker(mode='backtest')
    cerebro.broker = broker

    # # Analyzers
    # attach_analyzers(cerebro)

    print("Starting Backtest...")
    cerebro.run()
    cerebro.plot()