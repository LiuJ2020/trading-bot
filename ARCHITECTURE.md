# Trading System Architecture

**Version:** 1.0
**Last Updated:** 2025-12-19

## Table of Contents

1. [Design Principles](#design-principles)
2. [System Architecture](#system-architecture)
3. [Component Specifications](#component-specifications)
4. [Implementation Roadmap](#implementation-roadmap)
5. [Testing Strategy](#testing-strategy)
6. [Deployment & Operations](#deployment--operations)

---

## Design Principles

### Core Tenets

1. **Single Code Path**: The same strategy code runs in backtest, paper, and live modes
2. **Event-Driven**: All modes use event-driven architecture—no shortcuts
3. **Fast Iteration**: Research → backtest → paper → live in minutes, not days
4. **Production Realism**: Paper trading behaves identically to live trading
5. **Safe Promotion**: Config-driven deployment with instant rollback
6. **Failure Isolation**: Strategy failures don't cascade

### Key Innovation: Mode as Configuration

```python
# Same strategy code, different configuration
backtest_config = {
    "mode": "backtest",
    "clock": HistoricalClock,
    "data": HistoricalData,
    "execution": SimulatedExecution
}

live_config = {
    "mode": "live",
    "clock": RealtimeClock,
    "data": RealtimeData,
    "execution": BrokerExecution
}
```

---

## System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  Research Workspace                         │
│  • Jupyter notebooks for exploration                        │
│  • Feature prototyping & validation                         │
│  • Signal discovery & hypothesis testing                    │
│  OUTPUT: Feature definitions, strategy parameters           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Strategy SDK                             │
│  • Unified interface (on_start, on_market_event, etc.)      │
│  • No broker/clock dependencies                             │
│  • State management through context only                    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Simulation Engine                          │
│  • Mode-agnostic event loop                                 │
│  • Clock abstraction (historical/realtime)                  │
│  • Portfolio state tracking                                 │
│  • Event delivery to strategies                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Execution Engine                           │
│  • Order validation & risk checks                           │
│  • Slippage & latency simulation                            │
│  • Fill modeling (partial, market impact)                   │
│  • Broker adapter (paper vs live)                           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Broker / Exchange                          │
│  • Alpaca, Interactive Brokers, etc.                        │
│  • Paper trading endpoints                                  │
│  • Live trading endpoints                                   │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │
┌─────────────────────────────────────────────────────────────┐
│              Data & Feature Platform                        │
│  • Historical data store (immutable)                        │
│  • Realtime data feeds                                      │
│  • Feature store (versioned, reproducible)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Research Workspace

**Purpose**: Fast iteration on ideas without production constraints

**Components**:
- Jupyter environment for exploration
- Lightweight backtesting utilities
- Visualization tools
- Feature prototyping tools

**Outputs** (NOT executable trading code):
- Feature definitions
- Strategy hypotheses
- Parameter ranges
- Expected performance metrics

**File Structure**:
```
/research
  /notebooks
    exploration/
      signal_discovery.ipynb
      feature_analysis.ipynb
    validation/
      backtest_validation.ipynb
  /utils
    quick_backtest.py
    visualization.py
```

#### TODOs: Research Workspace

- [ ] **RW-001**: Set up Jupyter environment with common libraries (pandas, numpy, matplotlib, seaborn)
- [ ] **RW-002**: Create lightweight backtesting utility for notebooks (`QuickBacktest` class)
  - Should NOT be production code
  - Focus on speed over accuracy
  - Basic metrics calculation
- [ ] **RW-003**: Build feature exploration toolkit
  - Correlation analysis
  - Distribution visualization
  - Time-series stationarity tests
- [ ] **RW-004**: Create template notebooks for common workflows
  - Signal discovery template
  - Feature engineering template
  - Strategy validation template
- [ ] **RW-005**: Implement data loading utilities for notebooks
  - Easy access to historical data
  - Sample data for quick iteration
- [ ] **RW-006**: Build "research → production" checklist/workflow
  - Feature definition export format
  - Parameter validation
  - Documentation requirements

---

### 2. Strategy SDK

**Purpose**: Unified interface that runs identically in all modes

**Core Interface**:
```python
class Strategy(ABC):
    """Base class for all strategies."""

    @abstractmethod
    def on_start(self, context: Context) -> None:
        """Called once at strategy initialization."""
        pass

    @abstractmethod
    def on_market_event(self, event: MarketEvent, context: Context) -> None:
        """Called on each market data update."""
        pass

    @abstractmethod
    def generate_orders(self, context: Context) -> List[OrderIntent]:
        """Generate trading orders based on current state."""
        pass

    def on_order_update(self, order: Order, context: Context) -> None:
        """Called when order status changes (optional)."""
        pass

    def on_stop(self, context: Context) -> None:
        """Called when strategy is stopped (optional)."""
        pass
```

**Context Object** (Strategy's only window to the world):
```python
class Context:
    portfolio: Portfolio          # Current positions & cash
    features: FeatureStore        # Access to features
    current_time: datetime        # Current simulation time
    historical_data: DataAccess   # Historical market data
    metadata: Dict[str, Any]      # Strategy-specific state
    risk_limits: RiskLimits       # Position size limits, etc.
```

**Constraints**:
- ❌ No direct broker API calls
- ❌ No `datetime.now()` or time.time()
- ❌ No global state or singletons
- ❌ No file I/O from strategies
- ✅ All inputs via Context
- ✅ All outputs via OrderIntent

**File Structure**:
```
/strategies
  /sdk
    base.py                    # Strategy base class
    context.py                 # Context object
    events.py                  # Event types
    orders.py                  # OrderIntent, Order
    portfolio.py               # Portfolio state
  /implementations
    mean_reversion/
      strategy.py
      config.yaml
      README.md
    trend_following/
      strategy.py
      config.yaml
      README.md
```

#### TODOs: Strategy SDK

- [ ] **SDK-001**: Define `Strategy` base class with required methods
- [ ] **SDK-002**: Implement `Context` object with all necessary accessors
  - Portfolio state
  - Feature access
  - Historical data query
  - Time access (via clock)
- [ ] **SDK-003**: Define event types
  - `MarketEvent` (tick, bar, quote)
  - `OrderEvent` (filled, rejected, partial)
  - `PositionEvent` (opened, closed, modified)
- [ ] **SDK-004**: Create `OrderIntent` abstraction
  - Market, Limit, Stop orders
  - Time-in-force options
  - Quantity specification
- [ ] **SDK-005**: Implement `Portfolio` state object
  - Current positions
  - Cash balance
  - Buying power
  - PnL calculation
- [ ] **SDK-006**: Build strategy metadata system
  - Version tracking
  - Parameter schemas
  - Feature dependencies
  - Risk limit declarations
- [ ] **SDK-007**: Create strategy validation framework
  - Lint strategies for forbidden imports (datetime, requests, etc.)
  - Validate parameter schemas
  - Check feature dependencies exist
- [ ] **SDK-008**: Implement strategy registry
  - Dynamic loading
  - Version management
  - Config validation
- [ ] **SDK-009**: Create example strategies following best practices
  - Simple mean reversion
  - Moving average crossover
  - RSI-based strategy
- [ ] **SDK-010**: Write comprehensive strategy development guide
  - API documentation
  - Best practices
  - Common pitfalls

---

### 3. Simulation Engine (Core Innovation)

**Purpose**: Single event-driven engine for backtest, paper, and live

**Architecture**:
```python
class SimulationEngine:
    def __init__(
        self,
        clock: Clock,              # Historical or Realtime
        data_source: DataSource,   # Historical or Realtime
        execution: ExecutionEngine,# Simulated or Live
        strategies: List[Strategy]
    ):
        self.clock = clock
        self.data_source = data_source
        self.execution = execution
        self.strategies = strategies
        self.portfolio = Portfolio()

    def run(self):
        """Main event loop - same for all modes."""
        while not self.clock.is_done():
            current_time = self.clock.advance()

            # Get market events
            events = self.data_source.get_events(current_time)

            # Deliver to strategies
            for event in events:
                for strategy in self.strategies:
                    strategy.on_market_event(event, self.context)

            # Collect orders
            orders = []
            for strategy in self.strategies:
                orders.extend(strategy.generate_orders(self.context))

            # Execute
            for order in orders:
                self.execution.submit(order)

            # Process fills
            fills = self.execution.process_fills(current_time)
            self.portfolio.update(fills)
```

**Mode Configuration**:

| Component | Backtest | Paper | Live |
|-----------|----------|-------|------|
| Clock | `HistoricalClock` | `RealtimeClock` | `RealtimeClock` |
| Data | `HistoricalData` | `RealtimeData` | `RealtimeData` |
| Execution | `SimulatedExecution` | `SimulatedExecution` | `BrokerExecution` |
| Speed | Fast-forward | 1x | 1x |

**File Structure**:
```
/engine
  core/
    simulation_engine.py       # Main event loop
    clock.py                   # Clock abstractions
    context_builder.py         # Context construction
  clocks/
    historical_clock.py
    realtime_clock.py
  modes/
    backtest_config.py
    paper_config.py
    live_config.py
```

#### TODOs: Simulation Engine

- [ ] **SIM-001**: Implement `Clock` interface
  - `advance()` → datetime
  - `is_done()` → bool
  - `current_time()` → datetime
- [ ] **SIM-002**: Build `HistoricalClock`
  - Iterate through historical timestamps
  - Support variable speed (1x, 10x, max)
  - Handle market hours vs 24/7
- [ ] **SIM-003**: Build `RealtimeClock`
  - Wall-clock time
  - Sleep between ticks
  - Handle market hours
- [ ] **SIM-004**: Implement core event loop in `SimulationEngine`
  - Clock-driven advancement
  - Event collection from data source
  - Strategy invocation
  - Order collection
  - Execution processing
- [ ] **SIM-005**: Create `ContextBuilder`
  - Assemble Context from engine state
  - Provide feature access
  - Provide data access
  - Inject current time from clock
- [ ] **SIM-006**: Implement portfolio state tracking
  - Position updates from fills
  - Cash balance management
  - PnL calculation (realized & unrealized)
  - Mark-to-market updates
- [ ] **SIM-007**: Build event delivery system
  - Queue management
  - Priority handling (order events before market events)
  - Event timestamps
- [ ] **SIM-008**: Create mode configuration system
  - YAML/JSON config files
  - Environment selection (backtest/paper/live)
  - Component injection based on mode
- [ ] **SIM-009**: Implement strategy lifecycle management
  - on_start() invocation
  - on_stop() cleanup
  - Error handling & isolation
- [ ] **SIM-010**: Add simulation state persistence
  - Checkpoint/resume capability
  - State snapshots for debugging
- [ ] **SIM-011**: Build performance monitoring
  - Event processing latency
  - Strategy execution time
  - Memory usage tracking
- [ ] **SIM-012**: Create simulation replay system
  - Record events for debugging
  - Replay with different strategies
  - Step-through debugging mode

---

### 4. Execution Engine

**Purpose**: Realistic order modeling and broker integration

**Components**:

1. **Order Validation & Risk**
   - Position size limits
   - Buying power checks
   - Concentration limits
   - Regulatory constraints

2. **Slippage Models**
   ```python
   class SlippageModel(ABC):
       @abstractmethod
       def apply(self, order: Order, market: MarketState) -> Fill:
           pass

   class VolumeSlippage(SlippageModel):
       """Slippage based on order size vs volume."""

   class FixedSlippage(SlippageModel):
       """Fixed bps slippage."""
   ```

3. **Fill Modeling**
   - Market orders: immediate fill with slippage
   - Limit orders: price improvement, partial fills
   - Stop orders: trigger logic
   - Time-in-force handling

4. **Broker Adapter**
   ```python
   class BrokerAdapter(ABC):
       @abstractmethod
       def submit_order(self, order: Order) -> str:
           """Returns order ID."""
           pass

       @abstractmethod
       def get_fills(self) -> List[Fill]:
           """Poll for fills."""
           pass

       @abstractmethod
       def cancel_order(self, order_id: str) -> bool:
           pass
   ```

**File Structure**:
```
/execution
  core/
    execution_engine.py        # Main execution logic
    order_validator.py         # Risk checks
    fill_simulator.py          # Fill modeling
  slippage/
    base.py
    volume_slippage.py
    fixed_slippage.py
    market_impact.py
  adapters/
    simulated_broker.py        # For backtest/paper
    alpaca_adapter.py
    ibkr_adapter.py
  models/
    order.py                   # Order data structures
    fill.py                    # Fill data structures
```

#### TODOs: Execution Engine

- [ ] **EXE-001**: Define order lifecycle states
  - Pending, Submitted, PartiallyFilled, Filled, Rejected, Cancelled
- [ ] **EXE-002**: Implement `OrderValidator`
  - Position size limits
  - Cash availability
  - Leverage checks
  - Symbol restrictions
- [ ] **EXE-003**: Build `SlippageModel` interface and implementations
  - FixedSlippage (constant bps)
  - VolumeSlippage (order size / volume)
  - MarketImpactSlippage (more sophisticated)
- [ ] **EXE-004**: Create `FillSimulator`
  - Market order logic
  - Limit order matching
  - Stop order triggers
  - Partial fill simulation
- [ ] **EXE-005**: Implement latency simulation
  - Configurable delay between submit and fill
  - Network jitter modeling
- [ ] **EXE-006**: Build `BrokerAdapter` interface
  - submit_order()
  - cancel_order()
  - get_fills()
  - get_positions()
- [ ] **EXE-007**: Implement `SimulatedBroker` adapter
  - In-memory order book
  - Realistic fill logic
  - Same interface as live brokers
- [ ] **EXE-008**: Create Alpaca broker adapter
  - Order submission via API
  - Websocket for fills
  - Paper vs live endpoints
- [ ] **EXE-009**: Implement order reconciliation
  - Match expected vs actual fills
  - Detect missed fills
  - Handle broker rejections
- [ ] **EXE-010**: Build execution analytics
  - Slippage tracking
  - Fill rate analysis
  - Rejection reasons
- [ ] **EXE-011**: Create execution replay system
  - Record all orders & fills
  - Replay for debugging
  - Compare simulated vs actual
- [ ] **EXE-012**: Implement commission modeling
  - Per-share, per-order, tiered
  - Broker-specific models

---

### 5. Data & Feature Platform

**Purpose**: Unified data access across all modes

**Components**:

1. **Historical Data Store**
   - Immutable storage (Parquet, HDF5)
   - Symbol universe
   - OHLCV bars
   - Tick data (optional)
   - Corporate actions

2. **Realtime Data Feeds**
   - Websocket connections
   - Quote updates
   - Trade prints
   - Level 2 data (optional)

3. **Feature Store**
   ```python
   class Feature:
       name: str
       dependencies: List[str]      # Raw data needed
       window: int                  # Lookback period
       compute: Callable            # Computation function
       version: str                 # For reproducibility

   class FeatureStore:
       def get(self, feature: str, symbols: List[str], time: datetime):
           """Compute or retrieve feature values."""
           pass
   ```

**File Structure**:
```
/data
  sources/
    historical_data.py
    realtime_data.py
    broker_data.py            # Get data from broker
  storage/
    parquet_store.py
    hdf5_store.py
  loaders/
    yahoo_loader.py
    alpaca_loader.py
/features
  store/
    feature_store.py
    feature_registry.py
  definitions/
    technical.py              # RSI, MACD, etc.
    fundamental.py            # P/E, EPS, etc.
    custom.py                 # User-defined
  utils/
    caching.py
    versioning.py
```

#### TODOs: Data Platform

- [ ] **DATA-001**: Define `DataSource` interface
  - get_bars(symbols, start, end, timeframe)
  - get_latest_quote(symbols)
  - get_historical_ticks(symbols, start, end)
- [ ] **DATA-002**: Implement `HistoricalDataSource`
  - Read from Parquet/HDF5
  - Efficient time-range queries
  - Symbol filtering
- [ ] **DATA-003**: Build `RealtimeDataSource`
  - Websocket connection management
  - Quote buffering
  - Reconnection logic
- [ ] **DATA-004**: Create data ingestion pipeline
  - Download from Yahoo/Alpaca
  - Data validation
  - Storage in Parquet
  - Incremental updates
- [ ] **DATA-005**: Implement corporate actions handling
  - Splits
  - Dividends
  - Mergers
  - Price adjustment logic
- [ ] **DATA-006**: Build data quality checks
  - Missing data detection
  - Outlier detection
  - Timestamp validation
- [ ] **DATA-007**: Create universe management
  - Symbol lists
  - Sector/industry metadata
  - Market cap filters
  - Liquidity filters

#### TODOs: Feature Store

- [ ] **FEAT-001**: Define `Feature` class
  - Metadata (name, version, dependencies)
  - Computation function
  - Caching strategy
- [ ] **FEAT-002**: Implement `FeatureStore`
  - Feature registration
  - Dependency resolution
  - Lazy computation
  - Result caching
- [ ] **FEAT-003**: Build versioning system
  - Feature definition versions
  - Computation reproducibility
  - Schema evolution
- [ ] **FEAT-004**: Create common technical features
  - Returns (log, simple)
  - Volatility (std, ATR)
  - Momentum (RSI, MACD)
  - Trend (SMA, EMA)
- [ ] **FEAT-005**: Implement feature validation
  - Type checking
  - Range validation
  - NaN handling
- [ ] **FEAT-006**: Build feature computation engine
  - Vectorized operations
  - Incremental updates (for realtime)
  - Efficient lookback windows
- [ ] **FEAT-007**: Create feature testing framework
  - Unit tests for individual features
  - Property-based tests (stationarity, etc.)
  - Backtesting feature stability
- [ ] **FEAT-008**: Implement feature monitoring
  - Distribution drift detection
  - Correlation changes
  - Unexpected NaN rates

---

### 6. Control Plane

**Purpose**: Safe deployment, monitoring, and control

**Capabilities**:
- Enable/disable strategies
- Adjust capital allocation
- Pause execution
- Flatten all positions
- Emergency stop

**Components**:

1. **Strategy Manager**
   ```python
   class StrategyManager:
       def deploy(self, strategy: Strategy, config: Config):
           """Deploy strategy to environment."""

       def pause(self, strategy_id: str):
           """Pause strategy execution."""

       def resume(self, strategy_id: str):
           """Resume paused strategy."""

       def stop(self, strategy_id: str, flatten: bool = False):
           """Stop strategy, optionally flatten positions."""
   ```

2. **Monitoring**
   - PnL vs expected
   - Feature drift
   - Order fill rates
   - Latency tracking
   - Position concentration

3. **Alerting**
   - Threshold breaches
   - Unexpected behavior
   - System errors
   - Data quality issues

**File Structure**:
```
/control
  manager/
    strategy_manager.py
    deployment.py
    rollback.py
  monitoring/
    metrics_collector.py
    alerting.py
    dashboards.py
  config/
    environments.yaml          # dev, paper, prod
    risk_limits.yaml
    capital_allocation.yaml
```

#### TODOs: Control Plane

- [ ] **CTRL-001**: Build `StrategyManager`
  - Load strategies from config
  - Instantiate with correct mode
  - Lifecycle management
- [ ] **CTRL-002**: Implement deployment system
  - Config-driven deployment
  - Version tracking
  - Gradual rollout (canary)
- [ ] **CTRL-003**: Create rollback mechanism
  - Instant config revert
  - Position handling during rollback
  - State cleanup
- [ ] **CTRL-004**: Build monitoring dashboard
  - Real-time PnL
  - Position overview
  - Order flow
  - System health
- [ ] **CTRL-005**: Implement metrics collection
  - Time-series metrics storage
  - Strategy-level metrics
  - System-level metrics
- [ ] **CTRL-006**: Create alerting system
  - Threshold-based alerts
  - Anomaly detection
  - Multi-channel notifications (email, Slack)
- [ ] **CTRL-007**: Build feature drift monitoring
  - Compare current vs historical distributions
  - Alert on significant changes
- [ ] **CTRL-008**: Implement emergency controls
  - Kill switch
  - Flatten all positions
  - Pause all strategies
- [ ] **CTRL-009**: Create audit logging
  - All deployments
  - All manual interventions
  - All alerts
- [ ] **CTRL-010**: Build performance reporting
  - Daily PnL reports
  - Sharpe ratio tracking
  - Drawdown monitoring

---

### 7. Configuration Management

**Purpose**: Separate code from config, enable easy mode switching

**Config Structure**:

```yaml
# config/environments/backtest.yaml
mode: backtest

clock:
  type: historical
  start_date: 2023-01-01
  end_date: 2024-01-01
  speed: max

data:
  type: historical
  source: parquet
  path: /data/historical/

execution:
  type: simulated
  slippage:
    model: volume_based
    params:
      base_bps: 5
      volume_factor: 0.1
  latency:
    mean_ms: 10
    std_ms: 5

strategies:
  - name: mean_reversion_v1
    enabled: true
    capital: 100000
    risk_limits:
      max_position_size: 0.1  # 10% of capital
      max_leverage: 1.0

features:
  cache: true
  compute_on_demand: true
```

```yaml
# config/environments/paper.yaml
mode: paper

clock:
  type: realtime
  market_hours_only: true

data:
  type: realtime
  source: alpaca
  feed: iex

execution:
  type: simulated
  slippage:
    model: volume_based
    params:
      base_bps: 5
      volume_factor: 0.1
  latency:
    mean_ms: 50
    std_ms: 20

# ... same strategy config
```

```yaml
# config/environments/live.yaml
mode: live

clock:
  type: realtime
  market_hours_only: true

data:
  type: realtime
  source: alpaca
  feed: sip

execution:
  type: broker
  broker: alpaca
  endpoint: live  # or 'paper'

# ... strategy config with real capital limits
```

#### TODOs: Configuration

- [ ] **CFG-001**: Define configuration schema
  - Environment configs
  - Strategy configs
  - Risk limit configs
- [ ] **CFG-002**: Build config validation
  - Schema validation
  - Type checking
  - Required fields
- [ ] **CFG-003**: Implement config loader
  - YAML/JSON parsing
  - Environment variable substitution
  - Default values
- [ ] **CFG-004**: Create config templates
  - Backtest template
  - Paper template
  - Live template
- [ ] **CFG-005**: Build config versioning
  - Track config changes
  - Rollback capability
  - Audit trail
- [ ] **CFG-006**: Implement secrets management
  - API keys
  - Broker credentials
  - External service tokens

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goal**: Event-driven engine that can run simple backtests

**Milestones**:
1. Core event loop (SIM-004)
2. Historical clock (SIM-002)
3. Strategy SDK base classes (SDK-001, SDK-002)
4. Simple market events (SDK-003)
5. Portfolio state tracking (SIM-006, SDK-005)
6. Historical data source (DATA-002)
7. Config system (CFG-001, CFG-002, CFG-003)

**Deliverable**: Run a buy-and-hold strategy in backtest mode

**Success Criteria**:
- Event loop processes historical data
- Strategy receives market events
- Portfolio tracks positions correctly
- Config switches modes

---

### Phase 2: Realistic Execution (Weeks 3-4)

**Goal**: Accurate fill simulation and slippage modeling

**Milestones**:
1. Order intent & order models (SDK-004)
2. Execution engine core (EXE-001, EXE-002)
3. Slippage models (EXE-003)
4. Fill simulator (EXE-004)
5. Simulated broker adapter (EXE-007)
6. Order lifecycle (EXE-001)

**Deliverable**: Backtest with realistic slippage & fills

**Success Criteria**:
- Orders execute with slippage
- Partial fills work correctly
- Limit orders respect price levels
- Commission deducted

---

### Phase 3: Paper Trading (Weeks 5-6)

**Goal**: Realtime mode with simulated execution

**Milestones**:
1. Realtime clock (SIM-003)
2. Realtime data source (DATA-003)
3. Paper trading config (CFG-004)
4. Strategy isolation (SIM-009)
5. Monitoring basics (CTRL-004, CTRL-005)

**Deliverable**: Strategy runs in paper mode with live data

**Success Criteria**:
- Same strategy code as backtest
- Processes realtime data
- Orders simulated realistically
- No crashes for 24 hours

---

### Phase 4: Feature Platform (Weeks 7-8)

**Goal**: Reproducible, versioned features

**Milestones**:
1. Feature store (FEAT-001, FEAT-002)
2. Feature versioning (FEAT-003)
3. Common technical features (FEAT-004)
4. Feature validation (FEAT-005)
5. Integration with strategy SDK (SDK-002 update)

**Deliverable**: Strategies use features from feature store

**Success Criteria**:
- Features computed consistently
- Backtest vs paper features match
- Feature cache works
- Version changes don't break strategies

---

### Phase 5: Live Trading (Weeks 9-10)

**Goal**: Production deployment with real capital

**Milestones**:
1. Broker adapter (EXE-006, EXE-008)
2. Live execution mode (EXE-009)
3. Risk controls (EXE-002, CTRL-008)
4. Deployment system (CTRL-001, CTRL-002)
5. Alerting (CTRL-006)
6. Audit logging (CTRL-009)

**Deliverable**: First strategy live with real capital

**Success Criteria**:
- Orders reach broker
- Fills reconciled correctly
- Emergency stop works
- Alerts trigger correctly

---

### Phase 6: Operations & Scale (Weeks 11-12)

**Goal**: Multi-strategy operation, monitoring, safety

**Milestones**:
1. Strategy isolation (SIM-009 enhancement)
2. Multiple strategies running (CTRL-001)
3. Performance reporting (CTRL-010)
4. Feature drift monitoring (CTRL-007)
5. Rollback mechanism (CTRL-003)
6. Replay & debugging (SIM-012, EXE-011)

**Deliverable**: 3+ strategies running simultaneously

**Success Criteria**:
- Strategies don't interfere
- One failure doesn't stop others
- Monitoring dashboard complete
- Can replay any day's execution

---

## Testing Strategy

### Unit Tests

**Coverage Targets**:
- Strategy SDK: 100%
- Execution engine: 95%
- Feature store: 90%
- Data sources: 85%

**Key Test Areas**:
- Order lifecycle edge cases
- Slippage model accuracy
- Feature computation correctness
- Clock behavior (historical & realtime)
- Portfolio state updates
- Config validation

**TODO**:
- [ ] **TEST-001**: Set up pytest framework
- [ ] **TEST-002**: Create test fixtures for market data
- [ ] **TEST-003**: Build strategy test harness
- [ ] **TEST-004**: Write execution engine tests
- [ ] **TEST-005**: Implement feature store tests
- [ ] **TEST-006**: Create integration test suite

---

### Integration Tests

**Scenarios**:
1. **End-to-end backtest**
   - Load config → run strategy → verify results
2. **Paper trading simulation**
   - Simulate 1 day of trading
   - Verify event ordering
   - Check portfolio consistency
3. **Mode switching**
   - Same strategy in backtest, paper, live configs
   - Results should be consistent
4. **Feature consistency**
   - Compute features in backtest
   - Compute same features in paper
   - Values should match

**TODO**:
- [ ] **TEST-007**: Build end-to-end test framework
- [ ] **TEST-008**: Create deterministic data sets for testing
- [ ] **TEST-009**: Implement mode-switching tests
- [ ] **TEST-010**: Build feature consistency tests

---

### Property-Based Tests

Use Hypothesis for:
- Order validation (no invalid orders pass)
- Portfolio invariants (cash + positions = total value)
- Feature computation (no NaN/Inf unless expected)
- Event ordering (timestamps always increase)

**TODO**:
- [ ] **TEST-011**: Set up Hypothesis
- [ ] **TEST-012**: Write property tests for execution engine
- [ ] **TEST-013**: Write property tests for portfolio
- [ ] **TEST-014**: Write property tests for features

---

### Backtesting Validation

**Validation Strategy**:
1. Implement known strategies (buy-and-hold, 60/40)
2. Compare results to industry benchmarks
3. Verify no look-ahead bias
4. Test on multiple time periods

**Metrics to Validate**:
- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Average trade duration

**TODO**:
- [ ] **TEST-015**: Implement reference strategies
- [ ] **TEST-016**: Create validation data sets
- [ ] **TEST-017**: Build validation report generator
- [ ] **TEST-018**: Document validation results

---

## Deployment & Operations

### Environments

| Environment | Purpose | Capital | Risk |
|-------------|---------|---------|------|
| **dev** | Development & debugging | $0 (backtest only) | None |
| **paper** | Pre-production validation | $0 (simulated) | None |
| **prod** | Live trading | Real capital | Real |

### Promotion Criteria

**Paper → Prod Checklist**:
- [ ] Strategy runs for minimum 5 days in paper
- [ ] No crashes or errors
- [ ] Slippage within expected range (<10bps deviation)
- [ ] Fill rate >95%
- [ ] PnL within 1 std dev of backtest
- [ ] No risk limit violations
- [ ] Code review approved
- [ ] Monitoring & alerts configured

### Deployment Process

1. **Deploy to Paper**
   ```bash
   ./deploy.sh --strategy mean_reversion_v1 --env paper
   ```

2. **Monitor for N days**
   - Check dashboard daily
   - Review alerts
   - Compare to backtest

3. **Promote to Prod**
   ```bash
   ./deploy.sh --strategy mean_reversion_v1 --env prod --capital 10000
   ```

4. **Gradual Scale**
   - Start with small capital
   - Increase after stable period
   - Monitor continuously

### Rollback

**Instant Rollback**:
```bash
./rollback.sh --strategy mean_reversion_v1
```

**Actions**:
1. Stop strategy execution
2. Flatten positions (optional)
3. Revert to previous config
4. Send alert notification

### Monitoring

**Real-time Dashboards**:
- Strategy PnL
- Open positions
- Order flow
- Fill rates
- Latency metrics
- Feature values

**Alerts**:
- PnL threshold breaches
- Risk limit violations
- Execution failures
- Data quality issues
- System errors

**Daily Reports**:
- Strategy performance
- Trade log
- Slippage analysis
- Commission costs

---

## Appendix

### A. Repo Structure

```
trading-bot/
├── README.md
├── ARCHITECTURE.md              # This file
├── requirements.txt
├── setup.py
├── pyproject.toml
│
├── config/
│   ├── environments/
│   │   ├── backtest.yaml
│   │   ├── paper.yaml
│   │   └── live.yaml
│   ├── strategies/
│   │   ├── mean_reversion.yaml
│   │   └── trend_following.yaml
│   └── risk_limits.yaml
│
├── research/
│   ├── notebooks/
│   │   ├── exploration/
│   │   └── validation/
│   └── utils/
│       ├── quick_backtest.py
│       └── visualization.py
│
├── strategies/
│   ├── sdk/
│   │   ├── base.py
│   │   ├── context.py
│   │   ├── events.py
│   │   ├── orders.py
│   │   └── portfolio.py
│   └── implementations/
│       ├── mean_reversion/
│       └── trend_following/
│
├── engine/
│   ├── core/
│   │   ├── simulation_engine.py
│   │   ├── clock.py
│   │   └── context_builder.py
│   ├── clocks/
│   │   ├── historical_clock.py
│   │   └── realtime_clock.py
│   └── modes/
│       ├── backtest_config.py
│       ├── paper_config.py
│       └── live_config.py
│
├── execution/
│   ├── core/
│   │   ├── execution_engine.py
│   │   ├── order_validator.py
│   │   └── fill_simulator.py
│   ├── slippage/
│   │   ├── base.py
│   │   ├── volume_slippage.py
│   │   └── fixed_slippage.py
│   └── adapters/
│       ├── simulated_broker.py
│       └── alpaca_adapter.py
│
├── data/
│   ├── sources/
│   │   ├── historical_data.py
│   │   └── realtime_data.py
│   ├── storage/
│   │   └── parquet_store.py
│   └── loaders/
│       └── alpaca_loader.py
│
├── features/
│   ├── store/
│   │   ├── feature_store.py
│   │   └── feature_registry.py
│   ├── definitions/
│   │   ├── technical.py
│   │   └── custom.py
│   └── utils/
│       ├── caching.py
│       └── versioning.py
│
├── control/
│   ├── manager/
│   │   ├── strategy_manager.py
│   │   └── deployment.py
│   └── monitoring/
│       ├── metrics_collector.py
│       └── alerting.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── scripts/
    ├── deploy.sh
    ├── rollback.sh
    └── backtest.py
```

### B. Key Abstractions

**Separation of Concerns**:
- **Strategy**: Trading logic only
- **Engine**: Event delivery & orchestration
- **Execution**: Order handling & fills
- **Data**: Market data access
- **Features**: Computed signals
- **Control**: Deployment & monitoring

**Dependency Flow**:
```
Strategy → Context → Engine → Execution → Broker
                ↓
             Features → Data
```

**No Backward Dependencies**:
- Strategies don't know about engine
- Engine doesn't know about specific strategies
- Execution doesn't know about strategies
- Data doesn't know about features

### C. Common Pitfalls to Avoid

1. **Look-ahead bias in backtests**
   - Solution: Event-driven architecture prevents this

2. **Backtest overfitting**
   - Solution: Walk-forward validation, out-of-sample testing

3. **Paper trading unrealistic**
   - Solution: Slippage models, latency simulation

4. **Strategy drift** (backtest ≠ live code)
   - Solution: Same code path for all modes

5. **Hardcoded broker dependencies**
   - Solution: Broker adapters, dependency injection

6. **Global state in strategies**
   - Solution: Enforce stateless strategies, state in Context

7. **Manual deployment steps**
   - Solution: Config-driven deployment

8. **No rollback plan**
   - Solution: Config versioning, instant rollback

### D. Performance Considerations

**Backtest Speed**:
- Target: 1 year of daily data in <10 seconds
- Optimizations:
  - Vectorized feature computation
  - Efficient data structures (NumPy/Pandas)
  - Minimal event overhead
  - Feature caching

**Paper Trading Latency**:
- Target: <100ms strategy execution
- Optimizations:
  - Pre-computed features
  - Efficient event queues
  - Minimal allocations in hot path

**Memory**:
- Target: <2GB for single strategy
- Optimizations:
  - Sliding windows for features
  - Incremental computation
  - Data streaming (not loading all into memory)

### E. Security & Risk Management

**API Key Management**:
- Never commit keys to repo
- Use environment variables or secrets manager
- Rotate keys regularly

**Risk Limits** (Enforced in Code):
- Max position size per symbol
- Max total leverage
- Max daily loss (kill switch)
- Max order size
- Concentration limits

**Operational Risk**:
- Strategy isolation (one crash doesn't stop all)
- Graceful degradation (data outage handling)
- Automatic reconnection
- Circuit breakers

**Audit Trail**:
- Log all orders
- Log all fills
- Log all config changes
- Log all manual interventions

---

## TODO Summary by Priority

### P0 (Critical Path)

- [ ] SDK-001: Strategy base class
- [ ] SDK-002: Context object
- [ ] SIM-001: Clock interface
- [ ] SIM-002: Historical clock
- [ ] SIM-004: Core event loop
- [ ] DATA-002: Historical data source
- [ ] CFG-001, CFG-002, CFG-003: Config system
- [ ] SDK-004: OrderIntent
- [ ] EXE-001: Order lifecycle
- [ ] EXE-007: Simulated broker

### P1 (Near-term)

- [ ] SIM-003: Realtime clock
- [ ] DATA-003: Realtime data source
- [ ] EXE-003: Slippage models
- [ ] EXE-004: Fill simulator
- [ ] CTRL-004: Monitoring dashboard
- [ ] TEST-001 to TEST-006: Core testing

### P2 (Medium-term)

- [ ] FEAT-001, FEAT-002: Feature store
- [ ] EXE-008: Broker adapter (Alpaca)
- [ ] CTRL-001, CTRL-002: Deployment
- [ ] CTRL-006: Alerting
- [ ] TEST-007 to TEST-010: Integration tests

### P3 (Long-term)

- [ ] SIM-012: Replay system
- [ ] EXE-011: Execution replay
- [ ] CTRL-007: Feature drift monitoring
- [ ] CTRL-010: Performance reporting
- [ ] TEST-011 to TEST-018: Advanced testing

---

**End of Architecture Document**
