import backtrader as bt
from data.data_factory import get_data_feed
from broker.broker_factory import get_broker
from analyzers.analyzer_suite import attach_analyzers

def run_backtest(strategy_class):
    cerebro = bt.Cerebro()

    # Data
    data_feed = get_data_feed(mode='backtest')
    cerebro.adddata(data_feed)

    # Strategy
    cerebro.addstrategy(strategy_class)

    # Broker setup
    broker = get_broker(mode='backtest')
    cerebro.broker = broker

    # Analyzers
    attach_analyzers(cerebro)

    print("Starting Backtest...")
    cerebro.run()
    cerebro.plot()