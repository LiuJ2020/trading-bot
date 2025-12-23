"""Data quality validation for market data.

Performs comprehensive quality checks on OHLCV data:
- Completeness (missing bars, gaps)
- Consistency (OHLC relationships)
- Outliers (extreme price movements)
- Volume validation
- Timestamp validation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate market data quality.

    Performs multiple quality checks and generates a quality score.
    """

    def __init__(
        self,
        max_price_change_pct: float = 50.0,
        min_volume: int = 0,
        outlier_std_threshold: float = 5.0
    ):
        """Initialize data validator.

        Args:
            max_price_change_pct: Maximum allowed daily price change (%)
            min_volume: Minimum expected volume
            outlier_std_threshold: Std deviations for outlier detection
        """
        self.max_price_change_pct = max_price_change_pct
        self.min_volume = min_volume
        self.outlier_std_threshold = outlier_std_threshold

    def validate(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "1D"
    ) -> Dict:
        """Perform comprehensive validation on OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol ticker (for logging)
            timeframe: Data timeframe

        Returns:
            Dict with validation results and quality metrics
        """
        logger.info(f"Validating {len(df)} bars for {symbol} {timeframe}")

        if df.empty:
            return {
                'valid': False,
                'quality_score': 0.0,
                'errors': ['Empty DataFrame'],
                'metrics': {}
            }

        # Run all validation checks
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_bars': len(df),
            'issues': [],
            'warnings': [],
            'metrics': {}
        }

        # 1. Check required columns
        missing_cols = self._check_required_columns(df)
        if missing_cols:
            results['issues'].append(f"Missing columns: {missing_cols}")
            results['valid'] = False
            results['quality_score'] = 0.0
            return results

        # 2. Check for null values
        null_metrics = self._check_null_values(df)
        results['metrics'].update(null_metrics)
        if null_metrics['has_nulls']:
            results['warnings'].append(f"Found null values: {null_metrics['null_counts']}")

        # 3. Check OHLC relationships
        ohlc_metrics = self._check_ohlc_validity(df)
        results['metrics'].update(ohlc_metrics)
        if ohlc_metrics['invalid_ohlc'] > 0:
            results['warnings'].append(f"Invalid OHLC relationships: {ohlc_metrics['invalid_ohlc']} bars")

        # 4. Check for negative prices
        negative_metrics = self._check_negative_prices(df)
        results['metrics'].update(negative_metrics)
        if negative_metrics['negative_prices'] > 0:
            results['issues'].append(f"Found {negative_metrics['negative_prices']} negative prices")

        # 5. Check for zero volume
        volume_metrics = self._check_volume(df)
        results['metrics'].update(volume_metrics)
        if volume_metrics['zero_volume_bars'] > 0:
            results['warnings'].append(f"Zero volume: {volume_metrics['zero_volume_bars']} bars")

        # 6. Check for duplicates
        duplicate_metrics = self._check_duplicates(df)
        results['metrics'].update(duplicate_metrics)
        if duplicate_metrics['duplicate_bars'] > 0:
            results['warnings'].append(f"Duplicate timestamps: {duplicate_metrics['duplicate_bars']}")

        # 7. Check for gaps in time series
        gap_metrics = self._check_gaps(df, timeframe)
        results['metrics'].update(gap_metrics)
        if gap_metrics['missing_bars'] > 0:
            results['warnings'].append(f"Missing bars (gaps): {gap_metrics['missing_bars']}")

        # 8. Check for outliers
        outlier_metrics = self._check_outliers(df)
        results['metrics'].update(outlier_metrics)
        if outlier_metrics['outlier_bars'] > 0:
            results['warnings'].append(f"Outliers detected: {outlier_metrics['outlier_bars']} bars")

        # 9. Check timestamp ordering
        timestamp_metrics = self._check_timestamps(df)
        results['metrics'].update(timestamp_metrics)
        if not timestamp_metrics['timestamps_ordered']:
            results['issues'].append("Timestamps not in chronological order")

        # Calculate completeness percentage
        expected_bars = gap_metrics.get('expected_bars', len(df))
        actual_bars = len(df)
        completeness_pct = (actual_bars / expected_bars * 100) if expected_bars > 0 else 0
        results['metrics']['completeness_pct'] = completeness_pct

        # Calculate overall quality score (0-100)
        quality_score = self._calculate_quality_score(results['metrics'])
        results['quality_score'] = quality_score

        # Determine if data is valid (no critical issues)
        results['valid'] = len(results['issues']) == 0 and quality_score >= 50.0

        if results['valid']:
            logger.info(f"{symbol}: Quality score {quality_score:.1f}/100, {len(results['warnings'])} warnings")
        else:
            logger.warning(f"{symbol}: Validation failed. Issues: {results['issues']}")

        return results

    def _check_required_columns(self, df: pd.DataFrame) -> List[str]:
        """Check for required columns."""
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in required if col not in df.columns]

    def _check_null_values(self, df: pd.DataFrame) -> Dict:
        """Check for null values in OHLCV columns."""
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        null_counts = df[ohlcv_cols].isnull().sum()

        return {
            'has_nulls': null_counts.any(),
            'null_counts': null_counts[null_counts > 0].to_dict()
        }

    def _check_ohlc_validity(self, df: pd.DataFrame) -> Dict:
        """Check OHLC relationship validity."""
        # High should be >= Low
        # Close should be between Low and High
        # Open should be between Low and High
        invalid = (
            (df['high'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low'])
        )

        return {
            'invalid_ohlc': invalid.sum()
        }

    def _check_negative_prices(self, df: pd.DataFrame) -> Dict:
        """Check for negative prices."""
        price_cols = ['open', 'high', 'low', 'close']
        negative = (df[price_cols] < 0).any(axis=1)

        return {
            'negative_prices': negative.sum()
        }

    def _check_volume(self, df: pd.DataFrame) -> Dict:
        """Check volume data."""
        return {
            'zero_volume_bars': (df['volume'] == 0).sum(),
            'negative_volume': (df['volume'] < 0).sum()
        }

    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate timestamps."""
        duplicates = df.duplicated(subset=['timestamp'], keep=False)

        return {
            'duplicate_bars': duplicates.sum()
        }

    def _check_gaps(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Check for gaps in the time series."""
        if len(df) < 2:
            return {'missing_bars': 0, 'expected_bars': len(df)}

        df_sorted = df.sort_values('timestamp')
        timestamps = pd.to_datetime(df_sorted['timestamp'])

        # Determine expected frequency
        freq = self._timeframe_to_freq(timeframe)

        if freq is None:
            return {'missing_bars': 0, 'expected_bars': len(df)}

        # Generate expected date range
        start = timestamps.min()
        end = timestamps.max()

        try:
            # For daily data, only count business days
            if timeframe == '1D':
                expected_dates = pd.bdate_range(start, end, freq='B')
            else:
                expected_dates = pd.date_range(start, end, freq=freq)

            expected_bars = len(expected_dates)
            missing_bars = max(0, expected_bars - len(df))

            return {
                'missing_bars': missing_bars,
                'expected_bars': expected_bars
            }
        except Exception as e:
            logger.warning(f"Error checking gaps: {e}")
            return {'missing_bars': 0, 'expected_bars': len(df)}

    def _check_outliers(self, df: pd.DataFrame) -> Dict:
        """Check for price outliers using returns."""
        if len(df) < 2:
            return {'outlier_bars': 0}

        # Calculate returns
        returns = df['close'].pct_change()

        # Calculate z-scores
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0 or pd.isna(std_return):
            return {'outlier_bars': 0}

        z_scores = np.abs((returns - mean_return) / std_return)

        # Count outliers beyond threshold
        outliers = (z_scores > self.outlier_std_threshold).sum()

        return {
            'outlier_bars': outliers
        }

    def _check_timestamps(self, df: pd.DataFrame) -> Dict:
        """Check timestamp validity and ordering."""
        timestamps = pd.to_datetime(df['timestamp'])

        return {
            'timestamps_ordered': timestamps.is_monotonic_increasing,
            'timestamp_duplicates': timestamps.duplicated().sum()
        }

    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score (0-100).

        Quality score formula:
        - Start with 100
        - Deduct points for each issue
        - Weight by severity
        """
        score = 100.0
        total_bars = metrics.get('total_bars', 1)

        # Critical issues (heavy penalty)
        score -= metrics.get('negative_prices', 0) * 10
        if not metrics.get('timestamps_ordered', True):
            score -= 20

        # Major issues
        invalid_ohlc_pct = (metrics.get('invalid_ohlc', 0) / total_bars) * 100
        score -= invalid_ohlc_pct * 0.5

        duplicate_pct = (metrics.get('duplicate_bars', 0) / total_bars) * 100
        score -= duplicate_pct * 0.5

        # Minor issues
        missing_pct = (metrics.get('missing_bars', 0) / metrics.get('expected_bars', total_bars)) * 100
        score -= missing_pct * 0.2

        zero_volume_pct = (metrics.get('zero_volume_bars', 0) / total_bars) * 100
        score -= zero_volume_pct * 0.1

        outlier_pct = (metrics.get('outlier_bars', 0) / total_bars) * 100
        score -= outlier_pct * 0.1

        # Ensure score is between 0 and 100
        return max(0.0, min(100.0, score))

    def _timeframe_to_freq(self, timeframe: str) -> Optional[str]:
        """Convert timeframe to pandas frequency string."""
        mapping = {
            '1T': '1T',
            '5T': '5T',
            '15T': '15T',
            '30T': '30T',
            '1H': '1H',
            '1D': 'B',  # Business days
            '1W': 'W',
            '1M': 'M'
        }
        return mapping.get(timeframe)


def validate_and_clean(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict]:
    """Validate data and clean common issues.

    Args:
        df: DataFrame to validate and clean
        symbol: Symbol ticker

    Returns:
        Tuple of (cleaned_df, validation_results)
    """
    validator = DataValidator()

    # Run validation
    validation_results = validator.validate(df, symbol)

    # Clean data
    df_clean = df.copy()

    # Remove duplicate timestamps (keep last)
    df_clean = df_clean.drop_duplicates(subset=['timestamp'], keep='last')

    # Remove rows with negative prices
    price_cols = ['open', 'high', 'low', 'close']
    df_clean = df_clean[(df_clean[price_cols] >= 0).all(axis=1)]

    # Fill null values (forward fill for prices, 0 for volume)
    df_clean[price_cols] = df_clean[price_cols].ffill()
    df_clean['volume'] = df_clean['volume'].fillna(0)

    # Sort by timestamp
    df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)

    if len(df_clean) < len(df):
        logger.info(f"Cleaned {len(df) - len(df_clean)} problematic bars from {symbol}")

    return df_clean, validation_results
