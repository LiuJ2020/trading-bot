"""Portfolio state management.

Tracks positions, cash, and P&L.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from strategies.sdk.orders import OrderSide, Fill


@dataclass
class Position:
    """Individual position in a symbol."""
    symbol: str
    quantity: float  # Positive for long, negative for short
    avg_price: float
    opened_at: datetime
    last_updated: datetime
    realized_pnl: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def side(self) -> str:
        """Position side: 'long', 'short', or 'flat'."""
        if self.quantity > 0:
            return "long"
        elif self.quantity < 0:
            return "short"
        else:
            return "flat"

    @property
    def market_value(self) -> Optional[float]:
        """Market value (requires current price)."""
        # This will be calculated by Portfolio using current market prices
        return None

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L in dollars
        """
        if self.quantity == 0:
            return 0.0
        return (current_price - self.avg_price) * self.quantity

    def total_pnl(self, current_price: float) -> float:
        """Total P&L (realized + unrealized).

        Args:
            current_price: Current market price

        Returns:
            Total P&L in dollars
        """
        return self.realized_pnl + self.unrealized_pnl(current_price)


class Portfolio:
    """Portfolio state tracker.

    Manages:
    - Cash balance
    - Positions
    - P&L tracking
    - Buying power
    """

    def __init__(self, initial_cash: float, leverage: float = 1.0):
        """Initialize portfolio.

        Args:
            initial_cash: Starting cash balance
            leverage: Maximum leverage allowed (1.0 = no leverage)
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.leverage = leverage
        self.positions: Dict[str, Position] = {}
        self.closed_positions: list = []  # Historical positions
        self.trade_history: list = []  # All fills

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol.

        Args:
            symbol: Symbol to lookup

        Returns:
            Position object or None if no position
        """
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has position in symbol."""
        return symbol in self.positions and self.positions[symbol].quantity != 0

    def apply_fill(self, fill: Fill, timestamp: datetime) -> None:
        """Apply a fill to the portfolio.

        Updates positions and cash balance.

        Args:
            fill: Fill object
            timestamp: Fill timestamp
        """
        # Record in trade history
        self.trade_history.append(fill)

        # Calculate cash impact (negative for buys, positive for sells)
        cash_delta = -fill.quantity * fill.price if fill.side == OrderSide.BUY else fill.quantity * fill.price
        cash_delta -= fill.commission  # Commission always reduces cash

        self.cash += cash_delta

        # Update position
        symbol = fill.symbol
        signed_quantity = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity

        if symbol not in self.positions or self.positions[symbol].quantity == 0:
            # Opening new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=signed_quantity,
                avg_price=fill.price,
                opened_at=timestamp,
                last_updated=timestamp
            )
        else:
            # Modifying existing position
            position = self.positions[symbol]
            old_quantity = position.quantity
            new_quantity = old_quantity + signed_quantity

            if (old_quantity > 0 and new_quantity < 0) or (old_quantity < 0 and new_quantity > 0):
                # Position reversal - close old position first, then open new
                # Calculate realized P&L on close
                close_quantity = -old_quantity
                close_pnl = (fill.price - position.avg_price) * close_quantity
                position.realized_pnl += close_pnl

                # Store closed position
                closed = Position(
                    symbol=symbol,
                    quantity=old_quantity,
                    avg_price=position.avg_price,
                    opened_at=position.opened_at,
                    last_updated=timestamp,
                    realized_pnl=position.realized_pnl
                )
                self.closed_positions.append(closed)

                # Open new position in opposite direction
                remaining_quantity = new_quantity
                position.quantity = remaining_quantity
                position.avg_price = fill.price
                position.opened_at = timestamp
                position.realized_pnl = 0.0

            elif abs(new_quantity) < abs(old_quantity):
                # Partial close
                close_quantity = signed_quantity
                close_pnl = (fill.price - position.avg_price) * abs(close_quantity)
                position.realized_pnl += close_pnl
                position.quantity = new_quantity

            elif abs(new_quantity) > abs(old_quantity):
                # Adding to position - update average price
                total_cost = (position.avg_price * abs(old_quantity) +
                             fill.price * abs(signed_quantity))
                position.avg_price = total_cost / abs(new_quantity)
                position.quantity = new_quantity

            else:
                # Closing position exactly
                close_pnl = (fill.price - position.avg_price) * abs(old_quantity)
                position.realized_pnl += close_pnl
                position.quantity = 0.0

                # Store closed position
                closed = Position(
                    symbol=symbol,
                    quantity=old_quantity,
                    avg_price=position.avg_price,
                    opened_at=position.opened_at,
                    last_updated=timestamp,
                    realized_pnl=position.realized_pnl
                )
                self.closed_positions.append(closed)

            position.last_updated = timestamp

    def total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value.

        Args:
            current_prices: Dict of symbol -> current price

        Returns:
            Total portfolio value (cash + positions)
        """
        positions_value = sum(
            pos.quantity * current_prices.get(pos.symbol, pos.avg_price)
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate total unrealized P&L.

        Args:
            current_prices: Dict of symbol -> current price

        Returns:
            Unrealized P&L across all positions
        """
        return sum(
            pos.unrealized_pnl(current_prices.get(pos.symbol, pos.avg_price))
            for pos in self.positions.values()
        )

    def realized_pnl(self) -> float:
        """Calculate total realized P&L.

        Returns:
            Realized P&L across all positions (open and closed)
        """
        open_realized = sum(pos.realized_pnl for pos in self.positions.values())
        closed_realized = sum(pos.realized_pnl for pos in self.closed_positions)
        return open_realized + closed_realized

    def total_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate total P&L (realized + unrealized).

        Args:
            current_prices: Dict of symbol -> current price

        Returns:
            Total P&L
        """
        return self.realized_pnl() + self.unrealized_pnl(current_prices)

    def buying_power(self, current_prices: Dict[str, float]) -> float:
        """Calculate available buying power.

        Args:
            current_prices: Dict of symbol -> current price

        Returns:
            Available buying power considering leverage
        """
        total_value = self.total_value(current_prices)
        max_value = total_value * self.leverage
        used_value = abs(sum(
            pos.quantity * current_prices.get(pos.symbol, pos.avg_price)
            for pos in self.positions.values()
        ))
        return max_value - used_value

    def get_returns(self) -> float:
        """Get return since inception (percentage).

        Returns:
            Return as decimal (e.g., 0.15 = 15%)
        """
        if self.initial_cash == 0:
            return 0.0
        return (self.cash - self.initial_cash) / self.initial_cash

    def summary(self, current_prices: Dict[str, float]) -> dict:
        """Get portfolio summary.

        Args:
            current_prices: Dict of symbol -> current price

        Returns:
            Dictionary with portfolio metrics
        """
        total_val = self.total_value(current_prices)
        return {
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "total_value": total_val,
            "positions": len([p for p in self.positions.values() if p.quantity != 0]),
            "realized_pnl": self.realized_pnl(),
            "unrealized_pnl": self.unrealized_pnl(current_prices),
            "total_pnl": self.total_pnl(current_prices),
            "return_pct": ((total_val - self.initial_cash) / self.initial_cash * 100),
            "buying_power": self.buying_power(current_prices),
            "leverage": self.leverage
        }
