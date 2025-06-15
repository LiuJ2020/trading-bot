import backtrader as bt

def get_broker(mode='backtest'):
    broker = bt.brokers.BackBroker()
    broker.set_cash(100000)
    broker.setcommission(commission=0.001)
    return broker
