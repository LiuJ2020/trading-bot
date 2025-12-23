"""Context object - the strategy's window to the world.

All strategy interaction with the outside world goes through Context.
"""

from datetime import datetime
from typing import Any, Dict, Optional, List
import pandas as pd

from strategies.sdk.portfolio import Portfolio


class DataAccess:
    """Interface for accessing historical market data.

    Strategies use this to query past prices, volumes, etc.
    """

    def __init__(self, data_source):
        """Initialize data access.

        Args:
            data_source: Data source implementation
        """
        self._data_source = data_source

    def get_bars(
        self,
        symbols: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = "1D",
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get historical OHLCV bars.

        Args:
            symbols: List of symbols
            start: Start timestamp (None = from beginning)
            end: End timestamp (None = up to current time)
            timeframe: Bar timeframe (e.g., "1D", "1H", "5T")
            limit: Maximum number of bars per symbol

        Returns:
            DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        """
        return self._data_source.get_bars(symbols, start, end, timeframe, limit)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol.

        Args:
            symbol: Symbol to query

        Returns:
            Latest price or None
        """
        return self._data_source.get_latest_price(symbol)

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dict of symbol -> price
        """
        return {
            symbol: self.get_latest_price(symbol)
            for symbol in symbols
        }


class FeatureStore:
    """Interface for accessing computed features.

    Features are versioned, cached computations derived from raw data.
    """

    def __init__(self, feature_registry=None):
        """Initialize feature store.

        Args:
            feature_registry: Feature registry implementation
        """
        self._registry = feature_registry

    def get(
        self,
        feature_name: str,
        symbols: List[str],
        timestamp: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get feature values for symbols.

        Args:
            feature_name: Name of feature
            symbols: List of symbols
            timestamp: Point-in-time (None = latest)

        Returns:
            DataFrame with feature values
        """
        if self._registry is None:
            raise NotImplementedError("Feature store not configured")

        return self._registry.compute(feature_name, symbols, timestamp)

    def get_value(self, feature_name: str, symbol: str, timestamp: Optional[datetime] = None) -> Any:
        """Get single feature value.

        Args:
            feature_name: Name of feature
            symbol: Symbol
            timestamp: Point-in-time (None = latest)

        Returns:
            Feature value
        """
        df = self.get(feature_name, [symbol], timestamp)
        if df.empty:
            return None
        return df.iloc[0][feature_name]


class RiskLimits:
    """Risk limits enforced on strategy.

    These come from config and are enforced by the execution engine.
    """

    def __init__(
        self,
        max_position_size: float = 1.0,
        max_position_pct: float = 0.1,
        max_leverage: float = 1.0,
        max_concentration: float = 0.25,
        max_daily_loss: Optional[float] = None,
        allowed_symbols: Optional[List[str]] = None
    ):
        """Initialize risk limits.

        Args:
            max_position_size: Max position size in dollars
            max_position_pct: Max position size as % of portfolio
            max_leverage: Maximum leverage allowed
            max_concentration: Max % of portfolio in single symbol
            max_daily_loss: Max daily loss before circuit breaker (None = no limit)
            allowed_symbols: Whitelist of allowed symbols (None = all allowed)
        """
        self.max_position_size = max_position_size
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage
        self.max_concentration = max_concentration
        self.max_daily_loss = max_daily_loss
        self.allowed_symbols = allowed_symbols

    def can_trade_symbol(self, symbol: str) -> bool:
        """Check if symbol is allowed.

        Args:
            symbol: Symbol to check

        Returns:
            True if symbol can be traded
        """
        if self.allowed_symbols is None:
            return True
        return symbol in self.allowed_symbols


class Context:
    """Strategy context - the only interface strategies have to the world.

    Contains:
    - Portfolio state (positions, cash, P&L)
    - Data access (historical prices)
    - Feature access (computed features)
    - Time (current simulation time)
    - Risk limits
    - Strategy metadata (stateful information)
    """

    def __init__(
        self,
        portfolio: Portfolio,
        data_access: DataAccess,
        feature_store: FeatureStore,
        current_time: datetime,
        risk_limits: RiskLimits,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize context.

        Args:
            portfolio: Portfolio state
            data_access: Data access interface
            feature_store: Feature store interface
            current_time: Current simulation time
            risk_limits: Risk limits for strategy
            metadata: Strategy-specific state storage
        """
        self._portfolio = portfolio
        self._data = data_access
        self._features = feature_store
        self._current_time = current_time
        self._risk_limits = risk_limits
        self._metadata = metadata if metadata is not None else {}

    @property
    def portfolio(self) -> Portfolio:
        """Access portfolio state."""
        return self._portfolio

    @property
    def data(self) -> DataAccess:
        """Access historical data."""
        return self._data

    @property
    def features(self) -> FeatureStore:
        """Access feature store."""
        return self._features

    @property
    def current_time(self) -> datetime:
        """Get current simulation time.

        This is the ONLY way strategies should access time.
        Never use datetime.now() in strategies!
        """
        return self._current_time

    @property
    def risk_limits(self) -> RiskLimits:
        """Get risk limits."""
        return self._risk_limits

    @property
    def metadata(self) -> Dict[str, Any]:
        """Strategy-specific metadata storage.

        Use this to maintain strategy state across events.

        Example:
            # In on_start:
            context.metadata['indicators'] = {'sma_20': None}

            # In on_market_event:
            context.metadata['indicators']['sma_20'] = calculate_sma(...)
        """
        return self._metadata

    def get_latest_prices(self) -> Dict[str, float]:
        """Get latest prices for all positions + any other tracked symbols.

        Returns:
            Dict of symbol -> price
        """
        symbols = list(self.portfolio.positions.keys())
        if not symbols:
            return {}
        return self.data.get_latest_prices(symbols)
