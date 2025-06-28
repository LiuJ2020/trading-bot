import backtrader as bt

class BuyAndHoldStrategy(bt.Strategy):
    """
    Simple buy and hold strategy: buys at the first bar and holds until the end.
    Sizing is handled by the attached Sizer. Liquidates all positions at the last bar.
    """
    def __init__(self):
        self.bought = False

    def next(self):
        if not self.bought:
            for data in self.datas:
                self.buy(data=data)
            self.bought = True
