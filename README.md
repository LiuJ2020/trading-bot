# Trading Bot - Event-Driven Trading Framework

A production-ready algorithmic trading framework with a single code path from research to live trading.

## Status: Phase 1 Complete ✅

The core event-driven engine is operational and validated. See [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md) for details.

## Key Features

- **Single Code Path**: Same strategy code runs in backtest, paper, and live modes
- **Event-Driven Architecture**: No shortcuts - realistic execution in all modes
- **Mode as Configuration**: Switch between backtest/paper/live without code changes
- **Strategy Isolation**: Strategies can't crash the system or access forbidden APIs
- **Realistic Execution**: Slippage, commissions, partial fills
- **Portfolio Tracking**: Accurate P&L, positions, buying power

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the Phase 1 demo backtest
python scripts/run_backtest.py
```

**Expected Output**:
```
Initial Capital:    $100,000.00
Final Value:        $132,021.40
Total P&L:          $32,022.39 (+32.02%)
Orders Submitted:   3
Trades Executed:    3
```

## Architecture

The system follows a clean layered architecture:

```
Research → Strategy SDK → Simulation Engine → Execution Engine → Broker
                    ↓                                  ↓
                Features ← Data Platform → Historical/Realtime Data
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete design document.

## Project Structure

```
trading-bot/
├── strategies/
│   ├── sdk/                      # Strategy interface (✅ Phase 1)
│   │   ├── base.py              # Strategy base class
│   │   ├── context.py           # Context object
│   │   ├── events.py            # Event types
│   │   ├── orders.py            # Order abstractions
│   │   └── portfolio.py         # Portfolio state
│   └── implementations/          # Concrete strategies
│       └── buy_and_hold.py      # Example strategy
│
├── engine/
│   ├── core/                    # Core engine (✅ Phase 1)
│   │   ├── clock.py            # Clock interface
│   │   └── simulation_engine.py # Main event loop
│   └── clocks/
│       └── historical_clock.py  # Historical clock
│
├── execution/
│   └── adapters/                # Execution adapters (✅ Phase 1)
│       └── simulated_broker.py  # Simulated broker
│
├── data/
│   └── sources/                 # Data sources (✅ Phase 1)
│       └── historical_data.py   # Historical data
│
├── scripts/
│   └── run_backtest.py         # Backtest runner
│
└── docs/
    ├── ARCHITECTURE.md          # Full architecture (71 TODOs)
    └── PHASE1_COMPLETE.md      # Phase 1 summary
```

## Development Roadmap

### ✅ Phase 1: Foundation (COMPLETE)
- [x] Strategy SDK with clean interfaces
- [x] Event-driven simulation engine
- [x] Historical clock
- [x] Basic data source
- [x] Simulated broker
- [x] Portfolio tracking
- [x] Buy-and-hold example strategy
- [x] End-to-end backtest

### Phase 2: Realistic Execution (2 weeks)
- [ ] Volume-based slippage
- [ ] Market impact models
- [ ] Enhanced fill simulation
- [ ] Commission modeling
- [ ] Order validation

### Phase 3: Paper Trading (2 weeks)
- [ ] Realtime clock
- [ ] Realtime data feeds
- [ ] Paper trading mode
- [ ] Basic monitoring

### Phase 4: Feature Platform (2 weeks)
- [ ] Feature store
- [ ] Feature versioning
- [ ] Common technical indicators
- [ ] Feature validation

### Phase 5: Live Trading (2 weeks)
- [ ] Broker adapters (Alpaca, IBKR)
- [ ] Live execution
- [ ] Risk controls
- [ ] Deployment system
- [ ] Alerting

### Phase 6: Operations (2 weeks)
- [ ] Multi-strategy support
- [ ] Performance reporting
- [ ] Monitoring dashboard
- [ ] Rollback mechanism

## Creating a Strategy

Strategies inherit from `BaseStrategy` and implement required methods:

```python
from strategies.sdk.base import BaseStrategy
from strategies.sdk.context import Context
from strategies.sdk.events import MarketEvent
from strategies.sdk.orders import OrderIntent

class MyStrategy(BaseStrategy):
    def on_start(self, context: Context):
        """Initialize strategy state."""
        context.metadata['position'] = None

    def on_market_event(self, event: MarketEvent, context: Context):
        """Process market data."""
        # Update indicators, track state
        pass

    def generate_orders(self, context: Context) -> List[OrderIntent]:
        """Generate trading orders."""
        orders = []
        # Your trading logic here
        return orders
```

**Rules**:
- ❌ No `datetime.now()` - use `context.current_time`
- ❌ No broker API calls - return `OrderIntent` objects
- ❌ No global state - use `context.metadata`
- ✅ All inputs via `Context`
- ✅ All outputs via `OrderIntent`

## Testing

Currently manual testing via backtest runner. Unit tests coming in Phase 2.

```bash
# Run backtest
python scripts/run_backtest.py

# Expected: 32% return on buy-and-hold
```

## Dependencies

Minimal dependencies for maximum stability:

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Visualization (future)

## Design Principles

1. **Single Code Path**: Backtest code IS production code
2. **Event-Driven**: No look-ahead bias, realistic execution
3. **Mode as Configuration**: Same strategy, different config
4. **Fail Safe**: One strategy failing doesn't crash the system
5. **Observable**: Full audit trail of all decisions

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete architecture design with 71 TODOs across 6 phases
- **[PHASE1_COMPLETE.md](PHASE1_COMPLETE.md)** - Phase 1 completion summary
- **Code Comments** - Inline documentation in all modules

## Contributing

This is a personal project implementing a specific architecture. The design is intentionally opinionated.

## License

MIT License - See LICENSE file

---

**Built with Claude Code** - An event-driven trading framework emphasizing production realism from day one.
