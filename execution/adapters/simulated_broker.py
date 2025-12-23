"""Simulated broker for backtesting and paper trading.

Provides realistic order execution without hitting real brokers.
"""

from datetime import datetime
from typing import List, Dict, Optional
import random

from strategies.sdk.orders import (
    Order, OrderIntent, OrderStatus, OrderType, OrderSide, Fill
)
from strategies.sdk.events import QuoteEvent


class SimulatedBroker:
    """Simulated broker adapter.

    Simulates:
    - Order fills
    - Slippage
    - Partial fills (configurable)
    - Latency (configurable)
    - Rejections (rare)

    Used for both backtesting and paper trading.
    """

    def __init__(
        self,
        slippage_bps: float = 5.0,
        latency_ms: float = 10.0,
        commission_per_share: float = 0.0,
        partial_fill_probability: float = 0.0,
        rejection_probability: float = 0.001
    ):
        """Initialize simulated broker.

        Args:
            slippage_bps: Slippage in basis points (100 bps = 1%)
            latency_ms: Simulated latency in milliseconds
            commission_per_share: Commission per share
            partial_fill_probability: Probability of partial fills (0-1)
            rejection_probability: Probability of order rejection (0-1)
        """
        self.slippage_bps = slippage_bps
        self.latency_ms = latency_ms
        self.commission_per_share = commission_per_share
        self.partial_fill_probability = partial_fill_probability
        self.rejection_probability = rejection_probability

        self._orders: Dict[str, Order] = {}
        self._pending_fills: List[Fill] = []

    def submit_order(self, order: Order, current_price: float, timestamp: datetime) -> Order:
        """Submit order to simulated broker.

        Args:
            order: Order to submit
            current_price: Current market price
            timestamp: Submission timestamp

        Returns:
            Updated order object
        """
        # Check for random rejection
        if random.random() < self.rejection_probability:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "Simulated rejection"
            return order

        # Update status
        order.status = OrderStatus.SUBMITTED
        self._orders[order.order_id] = order

        # For market orders, fill immediately
        if order.order_type == OrderType.MARKET:
            self._fill_order(order, current_price, timestamp)
        else:
            # Limit/stop orders remain pending
            order.status = OrderStatus.ACCEPTED

        return order

    def _fill_order(
        self,
        order: Order,
        fill_price: float,
        timestamp: datetime,
        partial: bool = False
    ) -> None:
        """Fill an order.

        Args:
            order: Order to fill
            fill_price: Fill price
            timestamp: Fill timestamp
            partial: Whether this is a partial fill
        """
        # Apply slippage
        slippage_factor = self.slippage_bps / 10000
        if order.side == OrderSide.BUY:
            fill_price *= (1 + slippage_factor)
        else:
            fill_price *= (1 - slippage_factor)

        # Determine fill quantity
        remaining_qty = order.remaining_quantity or order.quantity
        if partial and random.random() < self.partial_fill_probability:
            # Partial fill (50-95% of remaining)
            fill_qty = remaining_qty * random.uniform(0.5, 0.95)
        else:
            # Full fill
            fill_qty = remaining_qty

        # Calculate commission
        commission = fill_qty * self.commission_per_share

        # Create fill
        fill = Fill.create(
            order=order,
            quantity=fill_qty,
            price=fill_price,
            timestamp=timestamp,
            commission=commission
        )

        # Update order
        order.filled_quantity += fill_qty
        order.remaining_quantity = order.quantity - order.filled_quantity
        order.fills.append(fill)

        # Update status
        if order.remaining_quantity == 0:
            order.status = OrderStatus.FILLED
            order.filled_at = timestamp
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        # Calculate average fill price
        total_cost = sum(f.price * f.quantity for f in order.fills)
        total_qty = sum(f.quantity for f in order.fills)
        order.avg_fill_price = total_cost / total_qty if total_qty > 0 else None

        # Add fill to pending queue
        self._pending_fills.append(fill)

    def process_market_update(self, price: float, symbol: str, timestamp: datetime) -> None:
        """Process market price update.

        Checks if any pending limit/stop orders should be filled.

        Args:
            price: Current market price
            symbol: Symbol
            timestamp: Update timestamp
        """
        for order in list(self._orders.values()):
            if order.status not in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
                continue

            if order.symbol != symbol:
                continue

            should_fill = False

            # Check limit orders
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= order.limit_price:
                    should_fill = True
                    fill_price = order.limit_price  # Price improvement
                elif order.side == OrderSide.SELL and price >= order.limit_price:
                    should_fill = True
                    fill_price = order.limit_price

            # Check stop orders
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and price >= order.stop_price:
                    should_fill = True
                    fill_price = price
                elif order.side == OrderSide.SELL and price <= order.stop_price:
                    should_fill = True
                    fill_price = price

            # TODO: Add stop-limit logic

            if should_fill:
                self._fill_order(order, fill_price, timestamp)

    def get_fills(self) -> List[Fill]:
        """Get pending fills and clear queue.

        Returns:
            List of fills since last call
        """
        fills = self._pending_fills.copy()
        self._pending_fills.clear()
        return fills

    def cancel_order(self, order_id: str, timestamp: datetime) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel
            timestamp: Cancellation timestamp

        Returns:
            True if cancelled, False if not found or can't cancel
        """
        order = self._orders.get(order_id)
        if not order:
            return False

        if not order.is_active():
            return False

        order.status = OrderStatus.CANCELLED
        order.cancelled_at = timestamp
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order or None
        """
        return self._orders.get(order_id)

    def get_active_orders(self) -> List[Order]:
        """Get all active orders.

        Returns:
            List of active orders
        """
        return [
            order for order in self._orders.values()
            if order.is_active()
        ]

    def reset(self) -> None:
        """Reset broker state (for backtesting)."""
        self._orders.clear()
        self._pending_fills.clear()

    def __repr__(self) -> str:
        active = len(self.get_active_orders())
        total = len(self._orders)
        return f"SimulatedBroker(active_orders={active}, total_orders={total})"
