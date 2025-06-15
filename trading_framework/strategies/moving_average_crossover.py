import backtrader as bt

class MovingAverageStrategy(bt.Strategy):
    def __init__(self):
        self.sma_fast = bt.indicators.SimpleMovingAverage(self.data.close, period=10)
        self.sma_slow = bt.indicators.SimpleMovingAverage(self.data.close, period=30)

    def next(self):
        if self.position.size == 0 and self.sma_fast[0] > self.sma_slow[0]:
            self.buy()
        elif self.position.size > 0 and self.sma_fast[0] < self.sma_slow[0]:
            self.sell()
