"""QuickBacktest - Lightweight backtesting for notebook workflows.

This is NOT production-quality backtesting. It's optimized for:
- Speed over accuracy
- Simple API
- Quick iteration on strategy ideas
- Basic metrics and visualization

For production backtesting, use the full engine/simulation framework.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class BacktestResult:
    """Results from a quick backtest.

    Attributes:
        equity_curve: Time series of portfolio value
        trades: DataFrame of individual trades
        metrics: Dict of performance metrics
        positions: Time series of position sizes
        returns: Daily returns series
    """

    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    positions: pd.DataFrame
    returns: pd.Series

    def __repr__(self) -> str:
        return (
            f"BacktestResult(\n"
            f"  Total Return: {self.metrics['total_return']:.2%}\n"
            f"  Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}\n"
            f"  Max Drawdown: {self.metrics['max_drawdown']:.2%}\n"
            f"  Win Rate: {self.metrics['win_rate']:.2%}\n"
            f"  Num Trades: {self.metrics['num_trades']:.0f}\n"
            f")"
        )

    def plot(self, figsize: tuple = (15, 10)) -> None:
        """Plot comprehensive backtest results.

        Args:
            figsize: Figure size (width, height)
        """
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('QuickBacktest Results', fontsize=16, fontweight='bold')

        # Equity curve
        ax = axes[0, 0]
        self.equity_curve.plot(ax=ax, linewidth=2, color='#2E86AB')
        ax.set_title('Equity Curve', fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)

        # Drawdown
        ax = axes[0, 1]
        drawdown = (self.equity_curve / self.equity_curve.cummax() - 1) * 100
        drawdown.plot(ax=ax, linewidth=2, color='#A23B72', alpha=0.7)
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='#A23B72')
        ax.set_title('Drawdown', fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)

        # Returns distribution
        ax = axes[1, 0]
        self.returns.hist(ax=ax, bins=50, alpha=0.7, color='#18A558', edgecolor='black')
        ax.axvline(self.returns.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_title('Returns Distribution', fontweight='bold')
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cumulative returns
        ax = axes[1, 1]
        cum_returns = (1 + self.returns).cumprod()
        cum_returns.plot(ax=ax, linewidth=2, color='#F18F01')
        ax.set_title('Cumulative Returns', fontweight='bold')
        ax.set_ylabel('Cumulative Return')
        ax.grid(True, alpha=0.3)

        # Trade analysis
        ax = axes[2, 0]
        if not self.trades.empty and 'pnl' in self.trades.columns:
            wins = self.trades[self.trades['pnl'] > 0]['pnl']
            losses = self.trades[self.trades['pnl'] <= 0]['pnl']

            ax.bar(['Wins', 'Losses'], [len(wins), len(losses)],
                   color=['#18A558', '#A23B72'], alpha=0.7, edgecolor='black')
            ax.set_title('Win/Loss Count', fontweight='bold')
            ax.set_ylabel('Number of Trades')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No trade data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Win/Loss Count', fontweight='bold')

        # Metrics table
        ax = axes[2, 1]
        ax.axis('off')
        metrics_text = '\n'.join([
            'Performance Metrics',
            '=' * 40,
            f"Total Return:      {self.metrics['total_return']:>10.2%}",
            f"Annual Return:     {self.metrics['annual_return']:>10.2%}",
            f"Sharpe Ratio:      {self.metrics['sharpe_ratio']:>10.2f}",
            f"Sortino Ratio:     {self.metrics['sortino_ratio']:>10.2f}",
            f"Max Drawdown:      {self.metrics['max_drawdown']:>10.2%}",
            f"Win Rate:          {self.metrics['win_rate']:>10.2%}",
            f"Profit Factor:     {self.metrics['profit_factor']:>10.2f}",
            f"Num Trades:        {self.metrics['num_trades']:>10.0f}",
            f"Avg Trade:         {self.metrics['avg_trade']:>10.2%}",
        ])
        ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes,
               fontfamily='monospace', fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()


class QuickBacktest:
    """Lightweight backtesting engine for notebooks.

    This is optimized for speed and simplicity, not accuracy.
    Use for rapid prototyping and idea validation only.

    Example:
        >>> def my_strategy(data, i):
        ...     if data['sma_fast'].iloc[i] > data['sma_slow'].iloc[i]:
        ...         return 1.0  # Long
        ...     return 0.0  # Flat
        >>>
        >>> bt = QuickBacktest(initial_capital=100000)
        >>> result = bt.run(data, my_strategy)
        >>> result.plot()
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 10 bps
        slippage: float = 0.0005,    # 5 bps
    ):
        """Initialize QuickBacktest.

        Args:
            initial_capital: Starting portfolio value
            commission: Commission as fraction (0.001 = 10 bps)
            slippage: Slippage as fraction (0.0005 = 5 bps)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(
        self,
        data: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame, int], float],
        symbol_col: str = 'close',
        verbose: bool = False,
    ) -> BacktestResult:
        """Run backtest with a strategy function.

        Args:
            data: DataFrame with OHLCV data and features (indexed by datetime)
            strategy_func: Function(data, i) -> position_size
                          Returns target position as fraction of capital (-1 to 1)
            symbol_col: Column name for prices (default 'close')
            verbose: Print progress

        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        if symbol_col not in data.columns:
            raise ValueError(f"Column '{symbol_col}' not found in data")

        # Initialize tracking
        n = len(data)
        equity = np.zeros(n)
        positions = np.zeros(n)
        trades = []

        cash = self.initial_capital
        position = 0.0  # Current position size in dollars
        last_price = 0.0

        for i in range(n):
            price = data[symbol_col].iloc[i]

            # Get target position from strategy
            target_fraction = strategy_func(data, i)
            target_fraction = np.clip(target_fraction, -1.0, 1.0)
            target_position = target_fraction * self.initial_capital

            # Calculate trade
            trade_size = target_position - position

            if abs(trade_size) > 0.01:  # Trade threshold
                # Apply slippage
                if trade_size > 0:
                    execution_price = price * (1 + self.slippage)
                else:
                    execution_price = price * (1 - self.slippage)

                # Execute trade
                trade_cost = abs(trade_size) * self.commission
                cash -= trade_size  # Negative trade_size = buy
                cash -= trade_cost
                position = target_position

                # Record trade
                trades.append({
                    'timestamp': data.index[i],
                    'price': execution_price,
                    'size': trade_size,
                    'commission': trade_cost,
                    'type': 'BUY' if trade_size > 0 else 'SELL',
                })

                if verbose and i % 50 == 0:
                    print(f"[{data.index[i].date()}] Trade: {trade_size/1000:.1f}k @ ${price:.2f}")

            # Update equity
            equity[i] = cash + position * (price / last_price if last_price > 0 else 1.0)
            positions[i] = position / price if price > 0 else 0
            last_price = price

            if verbose and i % 100 == 0:
                pct = (i / n) * 100
                print(f"Progress: {pct:.1f}% - Equity: ${equity[i]:,.0f}")

        # Convert results to pandas objects
        equity_series = pd.Series(equity, index=data.index, name='equity')
        positions_df = pd.DataFrame({'position': positions}, index=data.index)
        trades_df = pd.DataFrame(trades)

        # Calculate metrics
        metrics = self._calculate_metrics(equity_series, trades_df)

        # Calculate returns
        returns = equity_series.pct_change().fillna(0)

        return BacktestResult(
            equity_curve=equity_series,
            trades=trades_df,
            metrics=metrics,
            positions=positions_df,
            returns=returns,
        )

    def run_signals(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        symbol_col: str = 'close',
        position_size: float = 1.0,
        verbose: bool = False,
    ) -> BacktestResult:
        """Run backtest with pre-computed signals.

        Args:
            data: DataFrame with price data
            signals: Series with values -1 (short), 0 (flat), 1 (long)
            symbol_col: Column name for prices
            position_size: Position size as fraction of capital
            verbose: Print progress

        Returns:
            BacktestResult
        """
        signals_aligned = signals.reindex(data.index, fill_value=0)

        def strategy_func(df: pd.DataFrame, i: int) -> float:
            return signals_aligned.iloc[i] * position_size

        return self.run(data, strategy_func, symbol_col, verbose)

    def _calculate_metrics(
        self,
        equity: pd.Series,
        trades: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate performance metrics.

        Args:
            equity: Equity curve
            trades: Trades DataFrame

        Returns:
            Dict of metrics
        """
        # Returns
        returns = equity.pct_change().fillna(0)
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

        # Annualization factor (assuming daily data)
        days = (equity.index[-1] - equity.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Risk metrics
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)
        max_dd = self._calculate_max_drawdown(equity)

        # Trade metrics
        if not trades.empty and 'size' in trades.columns:
            trades_copy = trades.copy()
            # Calculate P&L per trade (simplified)
            if len(trades_copy) > 1:
                trades_copy['pnl'] = -trades_copy['size'].diff().fillna(0) * trades_copy['price']
                wins = trades_copy[trades_copy['pnl'] > 0]['pnl']
                losses = trades_copy[trades_copy['pnl'] < 0]['pnl']

                win_rate = len(wins) / len(trades_copy) if len(trades_copy) > 0 else 0
                profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 0
                avg_trade = trades_copy['pnl'].mean() / self.initial_capital if len(trades_copy) > 0 else 0
            else:
                win_rate = 0
                profit_factor = 0
                avg_trade = 0
            num_trades = len(trades_copy)
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade = 0
            num_trades = 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': num_trades,
            'avg_trade': avg_trade,
            'volatility': returns.std() * np.sqrt(252),
        }

    @staticmethod
    def _calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (annualized).

        Args:
            returns: Daily returns series
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)
        if excess_returns.std() == 0:
            return 0.0

        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    @staticmethod
    def _calculate_sortino(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (annualized).

        Args:
            returns: Daily returns series
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)

    @staticmethod
    def _calculate_max_drawdown(equity: pd.Series) -> float:
        """Calculate maximum drawdown.

        Args:
            equity: Equity curve

        Returns:
            Maximum drawdown as fraction
        """
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        return drawdown.min()
