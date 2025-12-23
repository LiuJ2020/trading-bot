"""Core simulation engine.

Single event-driven loop for backtest, paper, and live modes.
"""

from datetime import datetime
from typing import List, Dict, Optional
import logging

from engine.core.clock import Clock
from strategies.sdk.base import Strategy
from strategies.sdk.context import Context, DataAccess, FeatureStore, RiskLimits
from strategies.sdk.portfolio import Portfolio
from strategies.sdk.orders import Order, OrderIntent, OrderStatus
from strategies.sdk.events import BarEvent
from execution.adapters.simulated_broker import SimulatedBroker


logger = logging.getLogger(__name__)


class SimulationEngine:
    """Core simulation engine.

    This is the heart of the system - the same event loop runs in
    backtest, paper, and live modes.

    Responsibilities:
    - Advance time via clock
    - Fetch market events from data source
    - Deliver events to strategies
    - Collect orders from strategies
    - Submit orders to execution
    - Process fills and update portfolio
    - Build context for strategies
    """

    def __init__(
        self,
        clock: Clock,
        data_source,  # DataSource interface
        broker: SimulatedBroker,  # Or real broker adapter
        strategies: List[Strategy],
        initial_cash: float = 100000.0,
        leverage: float = 1.0,
        risk_limits: Optional[RiskLimits] = None
    ):
        """Initialize simulation engine.

        Args:
            clock: Clock instance (Historical or Realtime)
            data_source: Data source instance
            broker: Broker adapter (Simulated or real)
            strategies: List of strategies to run
            initial_cash: Initial portfolio cash
            leverage: Portfolio leverage
            risk_limits: Risk limits for strategies
        """
        self.clock = clock
        self.data_source = data_source
        self.broker = broker
        self.strategies = strategies

        # Portfolio state
        self.portfolio = Portfolio(initial_cash=initial_cash, leverage=leverage)

        # Risk limits
        self.risk_limits = risk_limits or RiskLimits(
            max_position_pct=0.1,
            max_leverage=leverage
        )

        # Context components
        self._data_access = DataAccess(data_source)
        self._feature_store = FeatureStore()  # Placeholder for now

        # Strategy metadata storage
        self._strategy_metadata: Dict[str, dict] = {
            strategy.name: {} for strategy in strategies
        }

        # Event counters
        self._event_count = 0
        self._order_count = 0

        # Current prices (for portfolio valuation)
        self._current_prices: Dict[str, float] = {}

    def run(self) -> Dict:
        """Run simulation.

        Returns:
            Dict with simulation results
        """
        logger.info(f"Starting simulation with {len(self.strategies)} strategies")

        # Advance clock to first timestamp to initialize time
        timestamp = self.clock.advance()
        if timestamp is None:
            raise ValueError("Clock has no timestamps")

        # Initialize strategies BEFORE processing any events
        for strategy in self.strategies:
            context = self._build_context(strategy.name)
            logger.info(f"Initializing strategy: {strategy.name}")
            strategy.on_start(context)

        # Main event loop
        while timestamp is not None:
            # Get market events for this timestamp
            events = self.data_source.get_events(timestamp)
            self._event_count += len(events)

            # Update current prices
            for event in events:
                if isinstance(event, BarEvent):
                    self._current_prices[event.symbol] = event.close

            # Deliver events to strategies
            for event in events:
                for strategy in self.strategies:
                    context = self._build_context(strategy.name)
                    try:
                        strategy.on_market_event(event, context)
                    except Exception as e:
                        logger.error(f"Error in {strategy.name}.on_market_event: {e}", exc_info=True)

            # Collect orders from strategies
            all_orders = []
            for strategy in self.strategies:
                context = self._build_context(strategy.name)
                try:
                    order_intents = strategy.generate_orders(context)
                    if order_intents:
                        all_orders.extend(order_intents)
                except Exception as e:
                    logger.error(f"Error in {strategy.name}.generate_orders: {e}", exc_info=True)

            # Process orders
            for intent in all_orders:
                self._process_order_intent(intent, timestamp)

            # Process any pending fills from broker
            fills = self.broker.get_fills()
            for fill in fills:
                self.portfolio.apply_fill(fill, timestamp)

            # Update broker with current prices (for limit/stop orders)
            for symbol, price in self._current_prices.items():
                self.broker.process_market_update(price, symbol, timestamp)

            # Log progress periodically
            if hasattr(self.clock, 'progress_percentage'):
                progress = self.clock.progress_percentage()
                if self._event_count % 1000 == 0:
                    pnl = self.portfolio.total_pnl(self._current_prices)
                    logger.info(f"Progress: {progress:.1f}%, Events: {self._event_count}, "
                              f"Orders: {self._order_count}, PnL: ${pnl:,.2f}")

            # Advance to next timestamp
            timestamp = self.clock.advance()

        # Stop strategies
        for strategy in self.strategies:
            context = self._build_context(strategy.name)
            try:
                strategy.on_stop(context)
            except Exception as e:
                logger.error(f"Error in {strategy.name}.on_stop: {e}", exc_info=True)

        # Return results
        return self._build_results()

    def _build_context(self, strategy_name: str) -> Context:
        """Build context for a strategy.

        Args:
            strategy_name: Name of strategy

        Returns:
            Context object
        """
        return Context(
            portfolio=self.portfolio,
            data_access=self._data_access,
            feature_store=self._feature_store,
            current_time=self.clock.current_time(),
            risk_limits=self.risk_limits,
            metadata=self._strategy_metadata[strategy_name]
        )

    def _process_order_intent(self, intent: OrderIntent, timestamp: datetime) -> None:
        """Process order intent from strategy.

        Args:
            intent: OrderIntent from strategy
            timestamp: Current timestamp
        """
        # Convert intent to order
        order = Order.from_intent(intent, created_at=timestamp)

        # Validate order
        if not self._validate_order(order):
            logger.warning(f"Order validation failed: {order}")
            return

        # Get current price for symbol
        current_price = self._current_prices.get(order.symbol)
        if current_price is None:
            logger.warning(f"No price data for {order.symbol}, rejecting order")
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "No price data"
            return

        # Submit to broker
        order = self.broker.submit_order(order, current_price, timestamp)
        self._order_count += 1

        logger.debug(f"Order submitted: {order.symbol} {order.side.value} {order.quantity} @ {current_price}")

    def _validate_order(self, order: Order) -> bool:
        """Validate order against risk limits.

        Args:
            order: Order to validate

        Returns:
            True if order passes validation
        """
        # Check if symbol is allowed
        if not self.risk_limits.can_trade_symbol(order.symbol):
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "Symbol not allowed"
            return False

        # Check buying power (simplified)
        current_price = self._current_prices.get(order.symbol, 0)
        order_value = order.quantity * current_price

        max_position_value = self.portfolio.total_value(self._current_prices) * self.risk_limits.max_position_pct

        if order_value > max_position_value:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = f"Order size {order_value:.2f} exceeds max position {max_position_value:.2f}"
            return False

        # Add more validation as needed
        return True

    def _build_results(self) -> Dict:
        """Build results dictionary.

        Returns:
            Dict with simulation metrics
        """
        portfolio_summary = self.portfolio.summary(self._current_prices)

        results = {
            "portfolio": portfolio_summary,
            "events_processed": self._event_count,
            "orders_submitted": self._order_count,
            "trades": len(self.portfolio.trade_history),
            "closed_positions": len(self.portfolio.closed_positions),
            "final_value": portfolio_summary["total_value"],
            "total_pnl": portfolio_summary["total_pnl"],
            "return_pct": portfolio_summary["return_pct"]
        }

        logger.info(f"Simulation complete: {results}")

        return results

    def get_portfolio(self) -> Portfolio:
        """Get portfolio state.

        Returns:
            Portfolio object
        """
        return self.portfolio

    def get_current_prices(self) -> Dict[str, float]:
        """Get current market prices.

        Returns:
            Dict of symbol -> price
        """
        return self._current_prices.copy()
