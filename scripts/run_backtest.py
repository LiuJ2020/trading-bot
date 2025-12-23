"""Simple backtest runner script.

Demonstrates Phase 1 implementation - running a buy-and-hold strategy.
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.core.simulation_engine import SimulationEngine
from engine.clocks.historical_clock import HistoricalClock
from data.sources.historical_data import HistoricalDataSource, generate_sample_data
from execution.adapters.simulated_broker import SimulatedBroker
from strategies.implementations.buy_and_hold import BuyAndHoldStrategy
from strategies.sdk.context import RiskLimits


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_simple_backtest():
    """Run a simple backtest with buy-and-hold strategy."""

    logger.info("=" * 80)
    logger.info("Phase 1 Backtest - Buy and Hold Strategy")
    logger.info("=" * 80)

    # Configuration
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    symbols = ["AAPL", "MSFT", "GOOGL"]
    initial_cash = 100000.0

    # Generate sample data
    logger.info(f"Generating sample data for {symbols}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    data = generate_sample_data(
        symbols=symbols,
        start=start_date,
        end=end_date,
        freq="1D",
        initial_price=100.0,
        volatility=0.02
    )

    logger.info(f"Generated {len(data)} bars")

    # Create components
    data_source = HistoricalDataSource(data, timeframe="1D")
    clock = HistoricalClock.from_data(data, time_column="timestamp", speed="max")
    broker = SimulatedBroker(
        slippage_bps=5.0,
        commission_per_share=0.001
    )

    # Create strategy
    strategy = BuyAndHoldStrategy(
        name="buy_and_hold",
        params={
            "symbols": symbols,
            # Equal weight allocation
        }
    )

    # Create risk limits
    risk_limits = RiskLimits(
        max_position_pct=0.4,  # Max 40% per position
        max_leverage=1.0
    )

    # Create engine
    engine = SimulationEngine(
        clock=clock,
        data_source=data_source,
        broker=broker,
        strategies=[strategy],
        initial_cash=initial_cash,
        leverage=1.0,
        risk_limits=risk_limits
    )

    # Run backtest
    logger.info("Starting backtest...")
    results = engine.run()

    # Print results
    logger.info("=" * 80)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Initial Capital:    ${initial_cash:,.2f}")
    logger.info(f"Final Value:        ${results['final_value']:,.2f}")
    logger.info(f"Total P&L:          ${results['total_pnl']:,.2f}")
    logger.info(f"Return:             {results['return_pct']:.2f}%")
    logger.info(f"Events Processed:   {results['events_processed']:,}")
    logger.info(f"Orders Submitted:   {results['orders_submitted']}")
    logger.info(f"Trades Executed:    {results['trades']}")
    logger.info(f"Closed Positions:   {results['closed_positions']}")
    logger.info("=" * 80)

    # Print portfolio details
    portfolio = engine.get_portfolio()
    logger.info("\nFinal Portfolio:")
    logger.info(f"  Cash: ${portfolio.cash:,.2f}")

    current_prices = engine.get_current_prices()
    logger.info(f"\nPositions:")
    for symbol, position in portfolio.positions.items():
        if position.quantity > 0:
            current_price = current_prices.get(symbol, position.avg_price)
            market_value = position.quantity * current_price
            pnl = position.unrealized_pnl(current_price)
            pnl_pct = (pnl / (position.quantity * position.avg_price)) * 100

            logger.info(f"  {symbol}:")
            logger.info(f"    Quantity:      {position.quantity:.0f}")
            logger.info(f"    Avg Price:     ${position.avg_price:.2f}")
            logger.info(f"    Current Price: ${current_price:.2f}")
            logger.info(f"    Market Value:  ${market_value:,.2f}")
            logger.info(f"    P&L:           ${pnl:,.2f} ({pnl_pct:+.2f}%)")

    return results


if __name__ == "__main__":
    try:
        results = run_simple_backtest()
        logger.info("\nBacktest completed successfully!")
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        sys.exit(1)
