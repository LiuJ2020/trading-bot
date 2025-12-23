"""Event types for market data and order updates.

All events are immutable and timestamped.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class EventType(Enum):
    """Types of events in the system."""
    BAR = "bar"              # OHLCV bar
    TICK = "tick"            # Individual trade
    QUOTE = "quote"          # Bid/ask quote
    ORDER = "order"          # Order status update
    POSITION = "position"    # Position update


@dataclass(frozen=True)
class Event:
    """Base event class."""
    timestamp: datetime
    event_type: EventType


@dataclass(frozen=True)
class MarketEvent(Event):
    """Base class for market data events."""
    symbol: str


@dataclass(frozen=True)
class BarEvent(MarketEvent):
    """OHLCV bar event.

    This is the most common event type for strategies.
    """
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = "1D"  # e.g., "1D", "1H", "5T"

    def __post_init__(self):
        object.__setattr__(self, 'event_type', EventType.BAR)


@dataclass(frozen=True)
class TickEvent(MarketEvent):
    """Individual trade event."""
    price: float
    size: float

    def __post_init__(self):
        object.__setattr__(self, 'event_type', EventType.TICK)


@dataclass(frozen=True)
class QuoteEvent(MarketEvent):
    """Bid/ask quote event."""
    bid: float
    ask: float
    bid_size: float
    ask_size: float

    def __post_init__(self):
        object.__setattr__(self, 'event_type', EventType.QUOTE)

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return (self.spread / self.mid) * 10000


@dataclass(frozen=True)
class OrderEvent(Event):
    """Order status update event."""
    order_id: str
    symbol: str
    status: str  # See execution.models.order.OrderStatus
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    message: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, 'event_type', EventType.ORDER)


@dataclass(frozen=True)
class PositionEvent(Event):
    """Position update event."""
    symbol: str
    quantity: float
    avg_price: float
    side: str  # "long" or "short"

    def __post_init__(self):
        object.__setattr__(self, 'event_type', EventType.POSITION)
