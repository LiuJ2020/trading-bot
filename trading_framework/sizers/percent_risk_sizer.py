import backtrader as bt

class PercentageRiskSizer(bt.Sizer):
    """
    Sizer that risks a fixed percentage of available cash per trade, based on stop loss distance.
    If no stop loss is set, it defaults to risking the percentage on the full position.
    """
    params = (
        ('risk_per_trade', 0.02),  # 2% of available cash
        ('stop_loss_perc', 0.05),  # 5% stop loss if not set by strategy
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            # sell everything if not buying
            position = self.broker.getposition(data)
            return position.size if position else 0
        price = data.close[0]
        risk_cash = cash * self.p.risk_per_trade
        # Try to get stop price from strategy, else use default
        stop_price = getattr(self.strategy, 'stop_price', None)
        if stop_price is not None:
            risk_per_share = abs(price - stop_price)
        else:
            risk_per_share = price * self.p.stop_loss_perc
        if risk_per_share == 0:
            return 0
        size = int(risk_cash / risk_per_share)
        return max(size, 0)
