"""Order abstractions for strategy SDK.

OrderIntent: What strategies create (intent to trade)
Order: What the execution engine manages (actual order state)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class OrderSide(Enum):
    """Order side (buy or sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    """Time in force specification."""
    DAY = "day"           # Good for day
    GTC = "gtc"           # Good til cancelled
    IOC = "ioc"           # Immediate or cancel
    FOK = "fok"           # Fill or kill


class OrderStatus(Enum):
    """Order lifecycle states."""
    PENDING = "pending"               # Created, not submitted yet
    SUBMITTED = "submitted"           # Submitted to broker/exchange
    ACCEPTED = "accepted"             # Accepted by broker
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderIntent:
    """Order intent created by strategies.

    This is what strategies return from generate_orders().
    The execution engine converts this to an Order.
    """
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    metadata: dict = field(default_factory=dict)  # Strategy-specific data

    def __post_init__(self):
        """Validate order intent."""
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")

        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders require limit_price")

        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop orders require stop_price")

        if self.order_type == OrderType.STOP_LIMIT:
            if self.limit_price is None or self.stop_price is None:
                raise ValueError("Stop-limit orders require both limit_price and stop_price")

    @classmethod
    def market_buy(cls, symbol: str, quantity: float, **kwargs) -> 'OrderIntent':
        """Convenience: create market buy order."""
        return cls(symbol=symbol, side=OrderSide.BUY, quantity=quantity,
                   order_type=OrderType.MARKET, **kwargs)

    @classmethod
    def market_sell(cls, symbol: str, quantity: float, **kwargs) -> 'OrderIntent':
        """Convenience: create market sell order."""
        return cls(symbol=symbol, side=OrderSide.SELL, quantity=quantity,
                   order_type=OrderType.MARKET, **kwargs)

    @classmethod
    def limit_buy(cls, symbol: str, quantity: float, limit_price: float, **kwargs) -> 'OrderIntent':
        """Convenience: create limit buy order."""
        return cls(symbol=symbol, side=OrderSide.BUY, quantity=quantity,
                   order_type=OrderType.LIMIT, limit_price=limit_price, **kwargs)

    @classmethod
    def limit_sell(cls, symbol: str, quantity: float, limit_price: float, **kwargs) -> 'OrderIntent':
        """Convenience: create limit sell order."""
        return cls(symbol=symbol, side=OrderSide.SELL, quantity=quantity,
                   order_type=OrderType.LIMIT, limit_price=limit_price, **kwargs)


@dataclass
class Order:
    """Actual order managed by execution engine.

    Created from OrderIntent after validation and risk checks.
    """
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    status: OrderStatus
    created_at: datetime
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    filled_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    avg_fill_price: Optional[float] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    broker_order_id: Optional[str] = None  # ID from broker/exchange
    metadata: dict = field(default_factory=dict)
    fills: list = field(default_factory=list)  # List of Fill objects

    def __post_init__(self):
        """Initialize remaining quantity if not set."""
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity

    @classmethod
    def from_intent(cls, intent: OrderIntent, created_at: datetime) -> 'Order':
        """Create Order from OrderIntent.

        Args:
            intent: OrderIntent from strategy
            created_at: Order creation timestamp

        Returns:
            Order object with PENDING status
        """
        return cls(
            order_id=str(uuid.uuid4()),
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            order_type=intent.order_type,
            status=OrderStatus.PENDING,
            created_at=created_at,
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            time_in_force=intent.time_in_force,
            metadata=intent.metadata.copy()
        )

    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        )

    def is_active(self) -> bool:
        """Check if order is active (can still be filled)."""
        return self.status in (
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED
        )

    @property
    def fill_percentage(self) -> float:
        """Percentage of order filled (0-100)."""
        return (self.filled_quantity / self.quantity) * 100 if self.quantity > 0 else 0


@dataclass
class Fill:
    """Individual fill for an order.

    An order can have multiple fills (partial fills).
    """
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(cls, order: Order, quantity: float, price: float,
               timestamp: datetime, commission: float = 0.0) -> 'Fill':
        """Create a fill for an order."""
        return cls(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            commission=commission
        )
