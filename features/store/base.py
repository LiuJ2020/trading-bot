"""Base feature class for all feature definitions.

Features are reusable signal/indicator definitions that work in both
research (notebooks) and production (backtests/live trading).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import numpy as np


@dataclass
class FeatureMetadata:
    """Metadata for feature versioning and reproducibility."""

    name: str
    version: str
    description: str
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "dependencies": self.dependencies,
            "parameters": self.parameters,
            "tags": list(self.tags)
        }


class Feature(ABC):
    """Base class for all features.

    Features are composable, cacheable signal computations that:
    - Have clear dependencies (required columns)
    - Specify lookback window requirements
    - Produce consistent output (Series with same index as input)
    - Handle edge cases gracefully (NaN for insufficient data)

    Example:
        ```python
        class RSI(Feature):
            def __init__(self, period: int = 14):
                super().__init__(
                    name=f"rsi_{period}",
                    version="1.0.0",
                    description=f"Relative Strength Index over {period} periods",
                    dependencies=["close"],
                    window=period + 1,
                    parameters={"period": period}
                )
                self.period = period

            def compute(self, data: pd.DataFrame) -> pd.Series:
                # Implementation
                pass
        ```
    """

    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        dependencies: List[str],
        window: int,
        parameters: Optional[Dict[str, Any]] = None,
        author: Optional[str] = None,
        tags: Optional[Set[str]] = None
    ):
        """Initialize feature.

        Args:
            name: Unique feature identifier (e.g., "rsi_14", "sma_50")
            version: Semantic version (e.g., "1.0.0") for reproducibility
            description: Human-readable description
            dependencies: List of required DataFrame columns (e.g., ["close"])
            window: Minimum rows needed for computation (lookback period)
            parameters: Feature-specific parameters for versioning
            author: Feature creator (optional)
            tags: Feature categorization tags (optional)
        """
        self.metadata = FeatureMetadata(
            name=name,
            version=version,
            description=description,
            author=author,
            created_at=datetime.now(),
            dependencies=dependencies,
            parameters=parameters or {},
            tags=tags or set()
        )
        self.window = window

    @property
    def name(self) -> str:
        """Feature name."""
        return self.metadata.name

    @property
    def version(self) -> str:
        """Feature version."""
        return self.metadata.version

    @property
    def dependencies(self) -> List[str]:
        """Required DataFrame columns."""
        return self.metadata.dependencies

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute feature values.

        Must handle edge cases:
        - Insufficient data (less than self.window rows)
        - Missing columns (dependencies not present)
        - NaN values in input data

        Args:
            data: DataFrame with OHLCV data and/or other features
                 Expected columns depend on self.dependencies

        Returns:
            Series with same index as input DataFrame
            Should return NaN for periods with insufficient data

        Raises:
            ValueError: If required dependencies are missing
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate that required dependencies are present.

        Args:
            data: Input DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        missing = set(self.dependencies) - set(data.columns)
        if missing:
            raise ValueError(
                f"Feature '{self.name}' requires columns {self.dependencies}, "
                f"but missing: {missing}"
            )

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        """Convenience method to compute feature.

        Args:
            data: Input DataFrame

        Returns:
            Computed feature Series
        """
        return self.compute(data)

    def __repr__(self) -> str:
        return (
            f"Feature(name='{self.name}', version='{self.version}', "
            f"dependencies={self.dependencies}, window={self.window})"
        )


class CompositeFeature(Feature):
    """Feature that depends on other features.

    Useful for building complex indicators from simpler ones.
    The FeatureStore will automatically resolve dependencies.

    Example:
        ```python
        class MACDSignal(CompositeFeature):
            def __init__(self):
                super().__init__(
                    name="macd_signal",
                    version="1.0.0",
                    description="MACD signal line",
                    feature_dependencies=["macd"],
                    window=26
                )

            def compute(self, data: pd.DataFrame) -> pd.Series:
                # Assumes 'macd' column is already computed
                return data["macd"].ewm(span=9).mean()
        ```
    """

    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        feature_dependencies: List[str],
        window: int,
        parameters: Optional[Dict[str, Any]] = None,
        author: Optional[str] = None,
        tags: Optional[Set[str]] = None
    ):
        """Initialize composite feature.

        Args:
            name: Feature name
            version: Feature version
            description: Feature description
            feature_dependencies: List of feature names this depends on
            window: Lookback window
            parameters: Feature parameters
            author: Feature author
            tags: Feature tags
        """
        super().__init__(
            name=name,
            version=version,
            description=description,
            dependencies=feature_dependencies,  # These are feature names, not columns
            window=window,
            parameters=parameters,
            author=author,
            tags=tags
        )
        self.feature_dependencies = feature_dependencies
