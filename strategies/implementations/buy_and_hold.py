"""Buy and hold strategy.

Simple strategy that buys on first day and holds.
Useful for testing and as a baseline.
"""

from typing import List

from strategies.sdk.base import BaseStrategy
from strategies.sdk.context import Context
from strategies.sdk.events import MarketEvent, BarEvent
from strategies.sdk.orders import OrderIntent, OrderSide


class BuyAndHoldStrategy(BaseStrategy):
    """Buy and hold strategy.

    Buys symbols on first bar and holds until end.

    Parameters:
        symbols: List of symbols to buy
        allocation_pct: Percent of capital to allocate per symbol (default: equal weight)
    """

    def __init__(self, name: str = "buy_and_hold", params: dict = None):
        """Initialize strategy.

        Args:
            name: Strategy name
            params: Strategy parameters
                - symbols: List of symbols to trade
                - allocation_pct: Optional dict of symbol -> allocation %
        """
        super().__init__(name, params)

        self.symbols = params.get("symbols", [])
        if not self.symbols:
            raise ValueError("buy_and_hold strategy requires 'symbols' parameter")

        # Equal weight by default
        self.allocation_pct = params.get("allocation_pct", {})
        if not self.allocation_pct:
            equal_weight = 1.0 / len(self.symbols)
            self.allocation_pct = {symbol: equal_weight for symbol in self.symbols}

    def on_start(self, context: Context) -> None:
        """Initialize strategy state.

        Args:
            context: Strategy context
        """
        # Track if we've made initial purchase
        context.metadata["initial_purchase_made"] = False
        context.metadata["symbols_to_buy"] = set(self.symbols)

    def on_market_event(self, event: MarketEvent, context: Context) -> None:
        """Process market event.

        We don't need to do anything here - all logic in generate_orders.

        Args:
            event: Market event
            context: Strategy context
        """
        # For buy-and-hold, we only react on first event
        # Mark this event as processed
        if isinstance(event, BarEvent) and event.symbol in context.metadata["symbols_to_buy"]:
            pass  # Will generate order in generate_orders()

    def generate_orders(self, context: Context) -> List[OrderIntent]:
        """Generate orders.

        Buy all symbols on first opportunity.

        Args:
            context: Strategy context

        Returns:
            List of order intents
        """
        orders = []

        # Only buy once at the start
        if context.metadata["initial_purchase_made"]:
            return orders

        # Check what we can buy
        symbols_to_buy = context.metadata["symbols_to_buy"]
        available_cash = context.portfolio.cash

        for symbol in symbols_to_buy:
            # Get current price
            current_price = context.data.get_latest_price(symbol)
            if current_price is None:
                continue  # Wait for price data

            # Calculate allocation
            allocation = self.allocation_pct.get(symbol, 0.0)
            target_value = available_cash * allocation

            # Calculate shares to buy
            shares = int(target_value / current_price)

            if shares > 0:
                # Create buy order
                order = OrderIntent.market_buy(
                    symbol=symbol,
                    quantity=shares,
                    metadata={
                        "strategy": self.name,
                        "allocation_pct": allocation
                    }
                )
                orders.append(order)

        # Mark as complete if we generated orders
        if orders:
            context.metadata["initial_purchase_made"] = True

        return orders

    def on_stop(self, context: Context) -> None:
        """Called when strategy stops.

        Args:
            context: Strategy context
        """
        # Print final positions
        print(f"\n{self.name} - Final Positions:")
        for symbol, position in context.portfolio.positions.items():
            if position.quantity > 0:
                current_price = context.data.get_latest_price(symbol)
                pnl = position.unrealized_pnl(current_price) if current_price else 0
                print(f"  {symbol}: {position.quantity} shares @ ${position.avg_price:.2f}, "
                      f"Current: ${current_price:.2f}, PnL: ${pnl:.2f}")
