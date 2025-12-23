"""Feature validation utilities.

Provides type checking, NaN handling, and data quality validation
for feature computations.
"""

from typing import Optional, Tuple, Dict, Any
import warnings

import pandas as pd
import numpy as np


class FeatureValidator:
    """Validates feature computation results."""

    @staticmethod
    def validate_series(
        series: pd.Series,
        name: str,
        expected_length: Optional[int] = None,
        allow_nan: bool = True,
        allow_inf: bool = False,
        min_valid_ratio: float = 0.0
    ) -> Tuple[bool, str]:
        """Validate a computed feature Series.

        Args:
            series: Feature Series to validate
            name: Feature name for error messages
            expected_length: Expected series length
            allow_nan: Whether NaN values are acceptable
            allow_inf: Whether infinite values are acceptable
            min_valid_ratio: Minimum ratio of valid (non-NaN) values required

        Returns:
            Tuple of (is_valid, error_message)
            error_message is empty string if valid
        """
        # Type check
        if not isinstance(series, pd.Series):
            return False, f"Feature '{name}' must return pd.Series, got {type(series)}"

        # Length check
        if expected_length is not None and len(series) != expected_length:
            return False, (
                f"Feature '{name}' has incorrect length: "
                f"expected {expected_length}, got {len(series)}"
            )

        # Check for infinite values
        if not allow_inf and np.isinf(series).any():
            inf_count = np.isinf(series).sum()
            return False, f"Feature '{name}' contains {inf_count} infinite value(s)"

        # Check NaN ratio
        nan_count = series.isna().sum()
        if len(series) > 0:
            nan_ratio = nan_count / len(series)
            valid_ratio = 1.0 - nan_ratio

            if not allow_nan and nan_count > 0:
                return False, f"Feature '{name}' contains {nan_count} NaN value(s)"

            if valid_ratio < min_valid_ratio:
                return False, (
                    f"Feature '{name}' has only {valid_ratio:.1%} valid values, "
                    f"minimum required is {min_valid_ratio:.1%}"
                )

        return True, ""

    @staticmethod
    def check_data_quality(
        data: pd.DataFrame,
        required_columns: list,
        min_rows: int = 1
    ) -> Tuple[bool, str]:
        """Check input data quality.

        Args:
            data: Input DataFrame
            required_columns: List of required column names
            min_rows: Minimum number of rows required

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check type
        if not isinstance(data, pd.DataFrame):
            return False, f"Data must be DataFrame, got {type(data)}"

        # Check length
        if len(data) < min_rows:
            return False, f"Insufficient data: {len(data)} rows, minimum {min_rows}"

        # Check columns
        missing = set(required_columns) - set(data.columns)
        if missing:
            return False, f"Missing required columns: {missing}"

        # Check for all-NaN columns
        for col in required_columns:
            if data[col].isna().all():
                return False, f"Column '{col}' contains only NaN values"

        return True, ""

    @staticmethod
    def warn_if_many_nans(
        series: pd.Series,
        name: str,
        threshold: float = 0.5
    ) -> None:
        """Issue warning if series has many NaN values.

        Args:
            series: Feature Series
            name: Feature name
            threshold: Warning threshold (ratio of NaN values)
        """
        if len(series) == 0:
            return

        nan_ratio = series.isna().sum() / len(series)
        if nan_ratio > threshold:
            warnings.warn(
                f"Feature '{name}' has {nan_ratio:.1%} NaN values "
                f"(threshold: {threshold:.1%})",
                UserWarning
            )

    @staticmethod
    def get_valid_range(series: pd.Series) -> Dict[str, Any]:
        """Get statistics about valid (non-NaN) values.

        Args:
            series: Feature Series

        Returns:
            Dictionary with statistics
        """
        valid_values = series.dropna()

        if len(valid_values) == 0:
            return {
                "count": 0,
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "valid_ratio": 0.0
            }

        return {
            "count": len(valid_values),
            "min": valid_values.min(),
            "max": valid_values.max(),
            "mean": valid_values.mean(),
            "std": valid_values.std(),
            "valid_ratio": len(valid_values) / len(series)
        }

    @staticmethod
    def validate_index_alignment(
        data: pd.DataFrame,
        result: pd.Series,
        name: str
    ) -> Tuple[bool, str]:
        """Verify result index matches input data index.

        Args:
            data: Input DataFrame
            result: Computed result Series
            name: Feature name

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not data.index.equals(result.index):
            return False, (
                f"Feature '{name}' result index does not match input data index"
            )
        return True, ""


def validate_ohlcv_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """Validate OHLCV data structure and relationships.

    Checks:
    - Required columns present
    - High >= Low
    - High >= Open, Close
    - Low <= Open, Close
    - Volume >= 0

    Args:
        data: DataFrame with OHLCV data

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = set(required_cols) - set(data.columns)

    if missing:
        return False, f"Missing OHLCV columns: {missing}"

    # Check high/low relationship
    if (data["high"] < data["low"]).any():
        invalid_count = (data["high"] < data["low"]).sum()
        return False, f"Found {invalid_count} rows where high < low"

    # Check high is highest
    if ((data["high"] < data["open"]) | (data["high"] < data["close"])).any():
        invalid_count = (
            (data["high"] < data["open"]) | (data["high"] < data["close"])
        ).sum()
        warnings.warn(
            f"Found {invalid_count} rows where high < open or high < close",
            UserWarning
        )

    # Check low is lowest
    if ((data["low"] > data["open"]) | (data["low"] > data["close"])).any():
        invalid_count = (
            (data["low"] > data["open"]) | (data["low"] > data["close"])
        ).sum()
        warnings.warn(
            f"Found {invalid_count} rows where low > open or low > close",
            UserWarning
        )

    # Check volume is non-negative
    if (data["volume"] < 0).any():
        invalid_count = (data["volume"] < 0).sum()
        return False, f"Found {invalid_count} rows with negative volume"

    return True, ""


def handle_edge_cases(
    series: pd.Series,
    replace_inf: bool = True,
    replace_value: float = np.nan,
    clip_range: Optional[Tuple[float, float]] = None
) -> pd.Series:
    """Handle common edge cases in feature computation.

    Args:
        series: Input Series
        replace_inf: Replace infinite values with replace_value
        replace_value: Value to use for replacement
        clip_range: Optional (min, max) tuple to clip values

    Returns:
        Cleaned Series
    """
    result = series.copy()

    # Replace infinities
    if replace_inf:
        result = result.replace([np.inf, -np.inf], replace_value)

    # Clip to range
    if clip_range is not None:
        result = result.clip(lower=clip_range[0], upper=clip_range[1])

    return result
