"""Base strategy class that all trading strategies must inherit from.

This enforces a unified interface across backtest, paper, and live modes.
"""

from abc import ABC, abstractmethod
from typing import List

from strategies.sdk.context import Context
from strategies.sdk.events import MarketEvent
from strategies.sdk.orders import OrderIntent, Order


class Strategy(ABC):
    """Base class for all trading strategies.

    All strategies must implement the required methods. Strategies are:
    - Stateless (state lives in Context.metadata)
    - Mode-agnostic (same code for backtest/paper/live)
    - Isolated (no global dependencies)

    Forbidden in strategies:
    - Direct broker API calls
    - datetime.now() or time.time()
    - Global state or singletons
    - File I/O
    - Network calls

    All external interaction happens through the Context object.
    """

    def __init__(self, name: str, params: dict = None):
        """Initialize strategy.

        Args:
            name: Unique strategy identifier
            params: Strategy-specific parameters (from config)
        """
        self.name = name
        self.params = params or {}

    @abstractmethod
    def on_start(self, context: Context) -> None:
        """Called once when strategy starts.

        Use this to:
        - Initialize strategy state in context.metadata
        - Validate parameters
        - Set up indicators

        Args:
            context: Strategy context with portfolio, features, data access
        """
        pass

    @abstractmethod
    def on_market_event(self, event: MarketEvent, context: Context) -> None:
        """Called on each market data update.

        Use this to:
        - Update indicators
        - Track market state
        - Update strategy state in context.metadata

        Do NOT generate orders here. Use generate_orders() for that.

        Args:
            event: Market data event (tick, bar, quote)
            context: Strategy context
        """
        pass

    @abstractmethod
    def generate_orders(self, context: Context) -> List[OrderIntent]:
        """Generate trading orders based on current state.

        Called after on_market_event. This is where trading logic lives.

        Returns:
            List of OrderIntent objects (can be empty)

        Args:
            context: Strategy context
        """
        pass

    def on_order_update(self, order: Order, context: Context) -> None:
        """Called when order status changes (optional).

        Use this to:
        - Track order fills
        - Adjust strategy state based on execution
        - Log order events

        Args:
            order: Order with updated status
            context: Strategy context
        """
        pass

    def on_stop(self, context: Context) -> None:
        """Called when strategy is stopped (optional).

        Use this to:
        - Clean up resources
        - Log final state
        - Flatten positions if needed

        Args:
            context: Strategy context
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class BaseStrategy(Strategy):
    """Convenience base class with common functionality.

    Provides default implementations for optional methods.
    Most strategies should inherit from this instead of Strategy.
    """

    def on_order_update(self, order: Order, context: Context) -> None:
        """Default implementation: do nothing."""
        pass

    def on_stop(self, context: Context) -> None:
        """Default implementation: do nothing."""
        pass
