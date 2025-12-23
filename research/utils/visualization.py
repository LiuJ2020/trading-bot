"""Visualization utilities for research notebooks.

Rich, publication-quality plots for strategy analysis.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#18A558',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'neutral': '#6C757D',
}


def plot_equity_curve(
    equity: Union[pd.Series, pd.DataFrame],
    benchmark: Optional[pd.Series] = None,
    title: str = 'Equity Curve',
    figsize: tuple = (12, 6),
    show_drawdown: bool = True,
) -> None:
    """Plot equity curve with optional benchmark and drawdown.

    Args:
        equity: Equity series or DataFrame with multiple strategies
        benchmark: Optional benchmark series
        title: Plot title
        figsize: Figure size (width, height)
        show_drawdown: Whether to show drawdown subplot

    Example:
        >>> plot_equity_curve(result.equity_curve, benchmark=spy_prices)
    """
    if show_drawdown:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    # Plot equity curves
    if isinstance(equity, pd.Series):
        ax1.plot(equity.index, equity.values, linewidth=2,
                label='Strategy', color=COLORS['primary'])
    else:
        for col in equity.columns:
            ax1.plot(equity.index, equity[col], linewidth=2, label=col)

    if benchmark is not None:
        # Normalize benchmark to start at same value
        if isinstance(equity, pd.Series):
            initial = equity.iloc[0]
        else:
            initial = equity.iloc[0, 0]

        benchmark_normalized = benchmark / benchmark.iloc[0] * initial
        ax1.plot(benchmark_normalized.index, benchmark_normalized.values,
                linewidth=2, linestyle='--', label='Benchmark',
                color=COLORS['neutral'], alpha=0.7)

    ax1.set_title(title, fontweight='bold', fontsize=14)
    ax1.set_ylabel('Portfolio Value ($)', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major')

    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Plot drawdown
    if show_drawdown:
        if isinstance(equity, pd.Series):
            dd = (equity / equity.cummax() - 1) * 100
            ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, color=COLORS['danger'])
            ax2.plot(dd.index, dd.values, linewidth=1.5, color=COLORS['danger'])
        else:
            for col in equity.columns:
                dd = (equity[col] / equity[col].cummax() - 1) * 100
                ax2.plot(dd.index, dd.values, linewidth=1.5, label=col)

        ax2.set_ylabel('Drawdown (%)', fontweight='bold')
        ax2.set_xlabel('Date', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_drawdown(
    equity: pd.Series,
    title: str = 'Drawdown Analysis',
    figsize: tuple = (12, 5),
) -> None:
    """Plot detailed drawdown analysis.

    Args:
        equity: Equity curve
        title: Plot title
        figsize: Figure size

    Example:
        >>> plot_drawdown(result.equity_curve)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Drawdown over time
    dd = (equity / equity.cummax() - 1) * 100
    ax1.fill_between(dd.index, dd.values, 0, alpha=0.3, color=COLORS['danger'])
    ax1.plot(dd.index, dd.values, linewidth=2, color=COLORS['danger'])
    ax1.set_title('Drawdown Over Time', fontweight='bold')
    ax1.set_ylabel('Drawdown (%)', fontweight='bold')
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Drawdown distribution
    ax2.hist(dd.values, bins=50, alpha=0.7, color=COLORS['danger'], edgecolor='black')
    ax2.axvline(dd.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {dd.mean():.2f}%')
    ax2.axvline(dd.min(), color='darkred', linestyle='--', linewidth=2, label=f'Max: {dd.min():.2f}%')
    ax2.set_title('Drawdown Distribution', fontweight='bold')
    ax2.set_xlabel('Drawdown (%)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_returns_distribution(
    returns: pd.Series,
    title: str = 'Returns Distribution',
    figsize: tuple = (12, 5),
) -> None:
    """Plot returns distribution with statistical analysis.

    Args:
        returns: Returns series
        title: Plot title
        figsize: Figure size

    Example:
        >>> plot_returns_distribution(result.returns)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram with normal distribution overlay
    ax1.hist(returns.dropna(), bins=50, alpha=0.7, density=True,
            color=COLORS['primary'], edgecolor='black', label='Returns')

    # Fit normal distribution
    mu, std = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax1.plot(x, (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std)**2),
            linewidth=2, color=COLORS['danger'], label='Normal Fit')

    ax1.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'Mean: {mu:.4f}')
    ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_title('Returns Distribution', fontweight='bold')
    ax1.set_xlabel('Daily Return', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(returns.dropna(), dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Test)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"""
    Mean: {mu:.4f}
    Std: {std:.4f}
    Skew: {returns.skew():.2f}
    Kurt: {returns.kurtosis():.2f}
    """
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(title, fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(
    data: pd.DataFrame,
    method: str = 'pearson',
    title: str = 'Correlation Matrix',
    figsize: tuple = (10, 8),
    annot: bool = True,
) -> None:
    """Plot correlation heatmap.

    Args:
        data: DataFrame with features
        method: Correlation method ('pearson', 'spearman', 'kendall')
        title: Plot title
        figsize: Figure size
        annot: Whether to annotate cells with values

    Example:
        >>> plot_correlation_matrix(data[['returns', 'volume', 'rsi_14']])
    """
    corr = data.corr(method=method)

    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, fmt='.2f')

    plt.title(title, fontweight='bold', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(
    data: pd.DataFrame,
    features: Optional[List[str]] = None,
    ncols: int = 3,
    figsize: tuple = (15, 10),
) -> None:
    """Plot distributions of multiple features.

    Args:
        data: DataFrame with features
        features: List of feature columns to plot (None = all numeric columns)
        ncols: Number of columns in subplot grid
        figsize: Figure size

    Example:
        >>> plot_feature_distributions(data, ['returns', 'rsi_14', 'volume_ratio'])
    """
    if features is None:
        features = data.select_dtypes(include=[np.number]).columns.tolist()

    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for idx, feature in enumerate(features):
        if idx >= len(axes):
            break

        ax = axes[idx]
        values = data[feature].dropna()

        if len(values) == 0:
            ax.text(0.5, 0.5, f'No data for\n{feature}', ha='center', va='center',
                   transform=ax.transAxes)
            continue

        # Histogram
        ax.hist(values, bins=50, alpha=0.7, color=COLORS['primary'], edgecolor='black')
        ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {values.mean():.3f}')
        ax.axvline(values.median(), color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {values.median():.3f}')

        ax.set_title(feature, fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(len(features), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Feature Distributions', fontweight='bold', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.show()


def plot_signals(
    data: pd.DataFrame,
    price_col: str = 'close',
    signal_col: Optional[str] = None,
    buy_signals: Optional[pd.Series] = None,
    sell_signals: Optional[pd.Series] = None,
    title: str = 'Trading Signals',
    figsize: tuple = (15, 8),
    show_features: Optional[List[str]] = None,
) -> None:
    """Plot price with trading signals and optional features.

    Args:
        data: DataFrame with price data
        price_col: Column name for price
        signal_col: Column name for continuous signals (-1 to 1)
        buy_signals: Boolean series indicating buy signals
        sell_signals: Boolean series indicating sell signals
        title: Plot title
        figsize: Figure size
        show_features: List of feature columns to plot in subplots

    Example:
        >>> plot_signals(data, buy_signals=data['buy'], sell_signals=data['sell'],
        ...             show_features=['rsi_14', 'volume_ratio'])
    """
    n_subplots = 1 + (1 if signal_col else 0) + (len(show_features) if show_features else 0)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_subplots, 1, height_ratios=[3] + [1] * (n_subplots - 1))

    # Price and signals
    ax_price = fig.add_subplot(gs[0])
    ax_price.plot(data.index, data[price_col], linewidth=2,
                 label='Price', color=COLORS['primary'])

    # Plot buy/sell signals
    if buy_signals is not None:
        buy_idx = buy_signals[buy_signals].index
        ax_price.scatter(buy_idx, data.loc[buy_idx, price_col],
                        marker='^', color=COLORS['success'], s=100,
                        label='Buy', zorder=5, edgecolors='black', linewidths=1)

    if sell_signals is not None:
        sell_idx = sell_signals[sell_signals].index
        ax_price.scatter(sell_idx, data.loc[sell_idx, price_col],
                        marker='v', color=COLORS['danger'], s=100,
                        label='Sell', zorder=5, edgecolors='black', linewidths=1)

    ax_price.set_title(title, fontweight='bold', fontsize=14)
    ax_price.set_ylabel('Price ($)', fontweight='bold')
    ax_price.legend(loc='upper left')
    ax_price.grid(True, alpha=0.3)

    subplot_idx = 1

    # Plot signal strength
    if signal_col:
        ax_signal = fig.add_subplot(gs[subplot_idx], sharex=ax_price)
        ax_signal.plot(data.index, data[signal_col], linewidth=2, color=COLORS['warning'])
        ax_signal.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax_signal.fill_between(data.index, data[signal_col], 0, alpha=0.3, color=COLORS['warning'])
        ax_signal.set_ylabel('Signal', fontweight='bold')
        ax_signal.grid(True, alpha=0.3)
        subplot_idx += 1

    # Plot features
    if show_features:
        for feature in show_features:
            if feature not in data.columns:
                continue

            ax_feat = fig.add_subplot(gs[subplot_idx], sharex=ax_price)
            ax_feat.plot(data.index, data[feature], linewidth=2, color=COLORS['secondary'])
            ax_feat.set_ylabel(feature, fontweight='bold')
            ax_feat.grid(True, alpha=0.3)
            subplot_idx += 1

    plt.xlabel('Date', fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    importance: Dict[str, float],
    title: str = 'Feature Importance',
    figsize: tuple = (10, 6),
    top_n: Optional[int] = None,
) -> None:
    """Plot feature importance as horizontal bar chart.

    Args:
        importance: Dict of feature_name -> importance_score
        title: Plot title
        figsize: Figure size
        top_n: Show only top N features

    Example:
        >>> importance = {'rsi_14': 0.35, 'volume_ratio': 0.25, 'sma_cross': 0.40}
        >>> plot_feature_importance(importance)
    """
    # Sort by importance
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    if top_n:
        sorted_features = sorted_features[:top_n]

    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(features, values, color=COLORS['primary'], alpha=0.7, edgecolor='black')

    # Color bars by importance
    max_val = max(values)
    for bar, val in zip(bars, values):
        if val > 0.7 * max_val:
            bar.set_color(COLORS['success'])
        elif val > 0.4 * max_val:
            bar.set_color(COLORS['warning'])
        else:
            bar.set_color(COLORS['neutral'])

    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()


def plot_pnl_attribution(
    trades: pd.DataFrame,
    group_by: str = 'symbol',
    title: str = 'P&L Attribution',
    figsize: tuple = (12, 6),
) -> None:
    """Plot P&L attribution by category.

    Args:
        trades: DataFrame with trades (must have 'pnl' column)
        group_by: Column to group by ('symbol', 'type', etc.)
        title: Plot title
        figsize: Figure size

    Example:
        >>> plot_pnl_attribution(result.trades, group_by='symbol')
    """
    if 'pnl' not in trades.columns or trades.empty:
        print("No P&L data available")
        return

    pnl_by_group = trades.groupby(group_by)['pnl'].sum().sort_values(ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bar chart
    colors = [COLORS['success'] if x > 0 else COLORS['danger'] for x in pnl_by_group.values]
    pnl_by_group.plot(kind='barh', ax=ax1, color=colors, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('P&L ($)', fontweight='bold')
    ax1.set_title(f'P&L by {group_by.capitalize()}', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Pie chart (only positive values)
    positive_pnl = pnl_by_group[pnl_by_group > 0]
    if len(positive_pnl) > 0:
        ax2.pie(positive_pnl.values, labels=positive_pnl.index, autopct='%1.1f%%',
               startangle=90, colors=sns.color_palette("Set3"))
        ax2.set_title('Profit Attribution', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No profitable trades', ha='center', va='center',
                transform=ax2.transAxes)

    fig.suptitle(title, fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
