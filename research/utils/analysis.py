"""Statistical analysis utilities for research notebooks.

Tools for analyzing features, testing hypotheses, and validating strategies.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import warnings


def calculate_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Calculate comprehensive performance metrics.

    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily, 12 for monthly)

    Returns:
        Dictionary of metrics

    Example:
        >>> metrics = calculate_metrics(result.returns)
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """
    returns_clean = returns.dropna()

    if len(returns_clean) == 0:
        return {}

    # Basic statistics
    mean_return = returns_clean.mean()
    std_return = returns_clean.std()
    total_return = (1 + returns_clean).prod() - 1

    # Annualized metrics
    annual_return = (1 + mean_return) ** periods_per_year - 1
    annual_vol = std_return * np.sqrt(periods_per_year)

    # Risk-adjusted returns
    excess_returns = returns_clean - (risk_free_rate / periods_per_year)
    sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
              if excess_returns.std() > 0 else 0)

    # Sortino ratio (downside deviation)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino = (excess_returns.mean() / downside_std * np.sqrt(periods_per_year)
               if downside_std > 0 else 0)

    # Calmar ratio (return / max drawdown)
    cum_returns = (1 + returns_clean).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

    # Higher moments
    skewness = returns_clean.skew()
    kurtosis = returns_clean.kurtosis()

    # Win/loss statistics
    winning_days = returns_clean[returns_clean > 0]
    losing_days = returns_clean[returns_clean < 0]
    win_rate = len(winning_days) / len(returns_clean) if len(returns_clean) > 0 else 0

    avg_win = winning_days.mean() if len(winning_days) > 0 else 0
    avg_loss = losing_days.mean() if len(losing_days) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Value at Risk (VaR)
    var_95 = returns_clean.quantile(0.05)
    cvar_95 = returns_clean[returns_clean <= var_95].mean()  # Conditional VaR

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'best_day': returns_clean.max(),
        'worst_day': returns_clean.min(),
    }


def correlation_analysis(
    data: pd.DataFrame,
    target: Optional[str] = None,
    method: str = 'pearson',
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Analyze correlations between features.

    Args:
        data: DataFrame with features
        target: Optional target column to correlate with
        method: Correlation method ('pearson', 'spearman', 'kendall')
        threshold: Minimum absolute correlation to report

    Returns:
        DataFrame of correlations sorted by strength

    Example:
        >>> corr = correlation_analysis(data, target='returns', threshold=0.3)
        >>> print(corr.head())
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if target:
        if target not in numeric_cols:
            raise ValueError(f"Target column '{target}' not found or not numeric")

        # Correlation with target
        correlations = data[numeric_cols].corr(method=method)[target].drop(target)
        result = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values,
            'abs_correlation': np.abs(correlations.values),
        })
    else:
        # All pairwise correlations
        corr_matrix = data[numeric_cols].corr(method=method)

        # Extract upper triangle
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j],
                    'abs_correlation': abs(corr_matrix.iloc[i, j]),
                })
        result = pd.DataFrame(pairs)

    # Filter by threshold and sort
    result = result[result['abs_correlation'] >= threshold]
    result = result.sort_values('abs_correlation', ascending=False).reset_index(drop=True)

    return result


def stationarity_test(
    series: pd.Series,
    test: str = 'adf',
    verbose: bool = True,
) -> Dict[str, Union[float, bool]]:
    """Test time series for stationarity.

    Args:
        series: Time series to test
        test: Test type ('adf' for Augmented Dickey-Fuller, 'kpss' for KPSS)
        verbose: Print results

    Returns:
        Dictionary with test results

    Example:
        >>> result = stationarity_test(data['returns'])
        >>> if result['is_stationary']:
        ...     print("Series is stationary")
    """
    series_clean = series.dropna()

    if len(series_clean) < 50:
        warnings.warn("Series too short for reliable stationarity test (< 50 observations)")

    if test.lower() == 'adf':
        # Augmented Dickey-Fuller test
        # H0: Series has unit root (non-stationary)
        # H1: Series is stationary
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(series_clean, autolag='AIC')
        adf_stat, p_value, lags, nobs, critical_values, icbest = result

        is_stationary = p_value < 0.05

        output = {
            'test': 'ADF',
            'test_statistic': adf_stat,
            'p_value': p_value,
            'lags_used': lags,
            'n_observations': nobs,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
        }

        if verbose:
            print(f"Augmented Dickey-Fuller Test Results:")
            print(f"  Test Statistic: {adf_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Lags Used: {lags}")
            print(f"  Critical Values:")
            for key, value in critical_values.items():
                print(f"    {key}: {value:.4f}")
            print(f"  Result: {'STATIONARY' if is_stationary else 'NON-STATIONARY'}")

    elif test.lower() == 'kpss':
        # KPSS test
        # H0: Series is stationary
        # H1: Series has unit root (non-stationary)
        from statsmodels.tsa.stattools import kpss

        result = kpss(series_clean, regression='c', nlags='auto')
        kpss_stat, p_value, lags, critical_values = result

        is_stationary = p_value > 0.05

        output = {
            'test': 'KPSS',
            'test_statistic': kpss_stat,
            'p_value': p_value,
            'lags_used': lags,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
        }

        if verbose:
            print(f"KPSS Test Results:")
            print(f"  Test Statistic: {kpss_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Lags Used: {lags}")
            print(f"  Critical Values:")
            for key, value in critical_values.items():
                print(f"    {key}: {value:.4f}")
            print(f"  Result: {'STATIONARY' if is_stationary else 'NON-STATIONARY'}")

    else:
        raise ValueError(f"Unknown test: {test}. Use 'adf' or 'kpss'")

    return output


def distribution_analysis(
    data: pd.Series,
    name: str = 'Series',
    verbose: bool = True,
) -> Dict[str, float]:
    """Analyze distribution of a series.

    Args:
        data: Data series to analyze
        name: Name of the series (for printing)
        verbose: Print results

    Returns:
        Dictionary with distribution statistics

    Example:
        >>> dist = distribution_analysis(data['returns'], name='Daily Returns')
    """
    data_clean = data.dropna()

    if len(data_clean) == 0:
        return {}

    # Basic statistics
    mean = data_clean.mean()
    median = data_clean.median()
    std = data_clean.std()
    skew = data_clean.skew()
    kurt = data_clean.kurtosis()

    # Quantiles
    q1 = data_clean.quantile(0.25)
    q3 = data_clean.quantile(0.75)
    iqr = q3 - q1

    # Outliers (using IQR method)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data_clean[(data_clean < lower_bound) | (data_clean > upper_bound)]
    outlier_pct = len(outliers) / len(data_clean) * 100

    # Normality test (Shapiro-Wilk)
    if len(data_clean) >= 3 and len(data_clean) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(data_clean)
        is_normal = shapiro_p > 0.05
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
        is_normal = None

    # Jarque-Bera test for normality
    jb_stat, jb_p = stats.jarque_bera(data_clean)
    is_normal_jb = jb_p > 0.05

    results = {
        'count': len(data_clean),
        'mean': mean,
        'median': median,
        'std': std,
        'skewness': skew,
        'kurtosis': kurt,
        'min': data_clean.min(),
        'max': data_clean.max(),
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'outlier_count': len(outliers),
        'outlier_pct': outlier_pct,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'jb_stat': jb_stat,
        'jb_p': jb_p,
        'is_normal': is_normal_jb,
    }

    if verbose:
        print(f"\nDistribution Analysis: {name}")
        print("=" * 60)
        print(f"Count:          {len(data_clean):>10,}")
        print(f"Mean:           {mean:>10.4f}")
        print(f"Median:         {median:>10.4f}")
        print(f"Std Dev:        {std:>10.4f}")
        print(f"Skewness:       {skew:>10.4f}  {'(left-skewed)' if skew < -0.5 else '(right-skewed)' if skew > 0.5 else '(symmetric)'}")
        print(f"Kurtosis:       {kurt:>10.4f}  {'(heavy-tailed)' if kurt > 1 else '(light-tailed)' if kurt < -1 else '(normal-tailed)'}")
        print(f"\nRange:")
        print(f"Min:            {data_clean.min():>10.4f}")
        print(f"Q1:             {q1:>10.4f}")
        print(f"Median:         {median:>10.4f}")
        print(f"Q3:             {q3:>10.4f}")
        print(f"Max:            {data_clean.max():>10.4f}")
        print(f"IQR:            {iqr:>10.4f}")
        print(f"\nOutliers:       {len(outliers):>10,}  ({outlier_pct:.2f}%)")
        print(f"\nNormality Tests:")
        if not np.isnan(shapiro_p):
            print(f"Shapiro-Wilk:   p={shapiro_p:.4f}  {'NORMAL' if is_normal else 'NOT NORMAL'}")
        print(f"Jarque-Bera:    p={jb_p:.4f}  {'NORMAL' if is_normal_jb else 'NOT NORMAL'}")

    return results


def feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'mutual_info',
    **kwargs,
) -> Dict[str, float]:
    """Calculate feature importance scores.

    Args:
        X: Feature DataFrame
        y: Target series
        method: Method to use ('mutual_info', 'correlation', 'random_forest')
        **kwargs: Additional arguments for the method

    Returns:
        Dictionary of feature -> importance score

    Example:
        >>> importance = feature_importance(features, returns, method='mutual_info')
        >>> from research.utils.visualization import plot_feature_importance
        >>> plot_feature_importance(importance)
    """
    # Remove rows with NaN in target
    valid_idx = ~y.isna()
    X_clean = X[valid_idx]
    y_clean = y[valid_idx]

    # Remove features with NaN
    X_clean = X_clean.dropna(axis=1)

    if len(X_clean) == 0 or len(X_clean.columns) == 0:
        warnings.warn("No valid data for feature importance calculation")
        return {}

    if method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression

        # Mutual information
        mi_scores = mutual_info_regression(X_clean, y_clean, **kwargs)
        importance = dict(zip(X_clean.columns, mi_scores))

    elif method == 'correlation':
        # Absolute correlation with target
        correlations = X_clean.corrwith(y_clean).abs()
        importance = correlations.to_dict()

    elif method == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor

        # Random forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, **kwargs)
        rf.fit(X_clean, y_clean)
        importance = dict(zip(X_clean.columns, rf.feature_importances_))

    else:
        raise ValueError(f"Unknown method: {method}. Use 'mutual_info', 'correlation', or 'random_forest'")

    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return importance


def rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Calculate rolling performance metrics.

    Args:
        returns: Returns series
        window: Rolling window size
        metrics: List of metrics to calculate. Options:
                'sharpe', 'sortino', 'vol', 'win_rate', 'max_dd'

    Returns:
        DataFrame with rolling metrics

    Example:
        >>> rolling = rolling_metrics(result.returns, window=60)
        >>> rolling['sharpe'].plot(title='Rolling Sharpe Ratio')
    """
    if metrics is None:
        metrics = ['sharpe', 'vol', 'win_rate']

    result = pd.DataFrame(index=returns.index)

    if 'sharpe' in metrics:
        rolling_sharpe = (
            returns.rolling(window).mean() / returns.rolling(window).std()
        ) * np.sqrt(252)
        result['sharpe'] = rolling_sharpe

    if 'sortino' in metrics:
        def calc_sortino(x):
            downside = x[x < 0].std()
            return (x.mean() / downside) * np.sqrt(252) if downside > 0 else 0

        result['sortino'] = returns.rolling(window).apply(calc_sortino, raw=False)

    if 'vol' in metrics:
        result['volatility'] = returns.rolling(window).std() * np.sqrt(252)

    if 'win_rate' in metrics:
        result['win_rate'] = returns.rolling(window).apply(
            lambda x: (x > 0).sum() / len(x), raw=False
        )

    if 'max_dd' in metrics:
        def calc_max_dd(x):
            cum = (1 + x).cumprod()
            running_max = cum.expanding().max()
            dd = (cum - running_max) / running_max
            return abs(dd.min())

        result['max_drawdown'] = returns.rolling(window).apply(calc_max_dd, raw=False)

    return result


def autocorrelation_analysis(
    series: pd.Series,
    lags: int = 20,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Analyze autocorrelation in a time series.

    Args:
        series: Time series to analyze
        lags: Number of lags to compute
        alpha: Significance level for confidence intervals

    Returns:
        DataFrame with autocorrelation values and significance

    Example:
        >>> acf_df = autocorrelation_analysis(data['returns'], lags=10)
        >>> print(acf_df[acf_df['significant']])
    """
    from statsmodels.tsa.stattools import acf

    series_clean = series.dropna()

    # Calculate ACF with confidence intervals
    acf_values, conf_int = acf(series_clean, nlags=lags, alpha=alpha, fft=False)

    # Create DataFrame
    result = pd.DataFrame({
        'lag': range(len(acf_values)),
        'acf': acf_values,
        'conf_lower': conf_int[:, 0] - acf_values,
        'conf_upper': conf_int[:, 1] - acf_values,
    })

    # Mark significant autocorrelations
    result['significant'] = (
        (result['acf'] < result['conf_lower']) |
        (result['acf'] > result['conf_upper'])
    )

    return result
