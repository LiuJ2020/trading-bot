"""FeatureStore for registering, computing, and caching features.

The FeatureStore is the central hub for feature computation in both
research and production environments.
"""

from typing import Dict, List, Optional, Set, Union
import logging
from collections import defaultdict
import warnings

import pandas as pd
import numpy as np

from features.store.base import Feature, CompositeFeature


logger = logging.getLogger(__name__)


class FeatureStore:
    """Central feature computation and caching engine.

    Features can be:
    - Registered programmatically or auto-discovered
    - Computed on-demand with automatic dependency resolution
    - Cached to avoid redundant computation
    - Versioned for reproducibility

    Example:
        ```python
        # Initialize store
        store = FeatureStore()

        # Register features
        store.register(RSI(period=14))
        store.register(SMA(period=50))

        # Compute features
        features = store.compute(["rsi_14", "sma_50"], data)
        # Returns DataFrame with columns: rsi_14, sma_50

        # Access individual feature
        rsi = store.compute_single("rsi_14", data)
        # Returns Series
        ```
    """

    def __init__(self, enable_cache: bool = True):
        """Initialize feature store.

        Args:
            enable_cache: Whether to cache computed features
        """
        self._features: Dict[str, Feature] = {}
        self._cache: Dict[str, pd.Series] = {}
        self._enable_cache = enable_cache
        self._computation_stats: Dict[str, int] = defaultdict(int)

    def register(self, feature: Feature) -> None:
        """Register a feature.

        Args:
            feature: Feature instance to register

        Raises:
            ValueError: If feature with same name already registered
        """
        if feature.name in self._features:
            existing = self._features[feature.name]
            if existing.version != feature.version:
                warnings.warn(
                    f"Replacing feature '{feature.name}' version "
                    f"{existing.version} with {feature.version}"
                )
            else:
                logger.debug(f"Feature '{feature.name}' already registered")
                return

        self._features[feature.name] = feature
        logger.info(
            f"Registered feature '{feature.name}' v{feature.version} "
            f"(dependencies: {feature.dependencies}, window: {feature.window})"
        )

    def register_batch(self, features: List[Feature]) -> None:
        """Register multiple features at once.

        Args:
            features: List of features to register
        """
        for feature in features:
            self.register(feature)

    def get_feature(self, name: str) -> Optional[Feature]:
        """Get feature by name.

        Args:
            name: Feature name

        Returns:
            Feature instance or None if not found
        """
        return self._features.get(name)

    def list_features(self, tag: Optional[str] = None) -> List[str]:
        """List all registered features.

        Args:
            tag: Optional tag filter

        Returns:
            List of feature names
        """
        if tag is None:
            return list(self._features.keys())
        else:
            return [
                name for name, feature in self._features.items()
                if tag in feature.metadata.tags
            ]

    def clear_cache(self) -> None:
        """Clear feature cache."""
        self._cache.clear()
        logger.debug("Feature cache cleared")

    def _resolve_dependencies(
        self,
        feature_names: List[str]
    ) -> List[str]:
        """Resolve feature dependencies in topological order.

        Args:
            feature_names: List of feature names to compute

        Returns:
            Ordered list of features to compute (dependencies first)

        Raises:
            ValueError: If feature not found or circular dependency detected
        """
        # Check all features exist
        for name in feature_names:
            if name not in self._features:
                raise ValueError(f"Feature '{name}' not registered")

        # Build dependency graph
        visited = set()
        order = []

        def visit(name: str, path: Set[str]) -> None:
            """DFS to detect cycles and determine order."""
            if name in path:
                raise ValueError(
                    f"Circular dependency detected: {' -> '.join(path)} -> {name}"
                )

            if name in visited:
                return

            feature = self._features[name]
            new_path = path | {name}

            # For composite features, resolve feature dependencies
            if isinstance(feature, CompositeFeature):
                for dep in feature.feature_dependencies:
                    if dep in self._features:
                        visit(dep, new_path)

            visited.add(name)
            order.append(name)

        for name in feature_names:
            visit(name, set())

        return order

    def _get_cache_key(self, feature_name: str, data: pd.DataFrame) -> str:
        """Generate cache key based on feature name and data hash.

        Args:
            feature_name: Feature name
            data: Input DataFrame

        Returns:
            Cache key string
        """
        # Use data shape and index hash as simple cache key
        # In production, you might want more sophisticated hashing
        data_hash = hash((
            id(data),  # Object identity for exact same DataFrame
            len(data),
            tuple(data.index[:5]) if len(data) > 0 else ()
        ))
        return f"{feature_name}_{data_hash}"

    def compute_single(
        self,
        feature_name: str,
        data: pd.DataFrame,
        use_cache: Optional[bool] = None
    ) -> pd.Series:
        """Compute a single feature.

        Args:
            feature_name: Name of feature to compute
            data: Input DataFrame with OHLCV data
            use_cache: Override default cache setting

        Returns:
            Series with computed feature values

        Raises:
            ValueError: If feature not registered or computation fails
        """
        if feature_name not in self._features:
            raise ValueError(f"Feature '{feature_name}' not registered")

        # Check cache
        use_cache = use_cache if use_cache is not None else self._enable_cache
        cache_key = self._get_cache_key(feature_name, data)

        if use_cache and cache_key in self._cache:
            logger.debug(f"Cache hit for '{feature_name}'")
            return self._cache[cache_key].copy()

        # Compute feature
        feature = self._features[feature_name]
        logger.debug(f"Computing feature '{feature_name}'")

        try:
            # For composite features, ensure dependencies are in data
            if isinstance(feature, CompositeFeature):
                # Compute dependencies first if not in data
                for dep_name in feature.feature_dependencies:
                    if dep_name not in data.columns:
                        data[dep_name] = self.compute_single(dep_name, data, use_cache)

            # Validate and compute
            feature.validate_data(data)
            result = feature.compute(data)

            # Track computation
            self._computation_stats[feature_name] += 1

            # Cache result
            if use_cache:
                self._cache[cache_key] = result.copy()

            return result

        except Exception as e:
            logger.error(f"Error computing feature '{feature_name}': {e}")
            raise

    def compute(
        self,
        feature_names: Union[str, List[str]],
        data: pd.DataFrame,
        use_cache: Optional[bool] = None
    ) -> pd.DataFrame:
        """Compute multiple features with dependency resolution.

        Args:
            feature_names: Single feature name or list of feature names
            data: Input DataFrame with OHLCV data
            use_cache: Override default cache setting

        Returns:
            DataFrame with computed features as columns

        Raises:
            ValueError: If any feature fails to compute
        """
        # Handle single feature name
        if isinstance(feature_names, str):
            feature_names = [feature_names]

        # Resolve dependencies
        ordered_features = self._resolve_dependencies(feature_names)

        # Make a copy of data to avoid modifying input
        result_data = data.copy()

        # Compute features in dependency order
        for name in ordered_features:
            if name not in result_data.columns:
                result_data[name] = self.compute_single(name, result_data, use_cache)

        # Return only requested features
        return result_data[feature_names]

    def get_stats(self) -> Dict[str, int]:
        """Get computation statistics.

        Returns:
            Dictionary mapping feature names to computation count
        """
        return dict(self._computation_stats)

    def get_info(self, feature_name: str) -> dict:
        """Get detailed information about a feature.

        Args:
            feature_name: Feature name

        Returns:
            Dictionary with feature metadata

        Raises:
            ValueError: If feature not found
        """
        if feature_name not in self._features:
            raise ValueError(f"Feature '{feature_name}' not registered")

        feature = self._features[feature_name]
        info = feature.metadata.to_dict()
        info["window"] = feature.window
        info["computed_count"] = self._computation_stats.get(feature_name, 0)

        return info

    def validate_features(
        self,
        feature_names: List[str],
        data: pd.DataFrame
    ) -> Dict[str, bool]:
        """Validate that features can be computed on given data.

        Args:
            feature_names: List of feature names
            data: Input DataFrame

        Returns:
            Dictionary mapping feature names to validation status (True/False)
        """
        results = {}

        for name in feature_names:
            try:
                if name not in self._features:
                    results[name] = False
                    continue

                feature = self._features[name]

                # Check if we have enough data
                if len(data) < feature.window:
                    results[name] = False
                    continue

                # Check dependencies
                try:
                    feature.validate_data(data)
                    results[name] = True
                except ValueError:
                    results[name] = False

            except Exception:
                results[name] = False

        return results

    def __repr__(self) -> str:
        return (
            f"FeatureStore(features={len(self._features)}, "
            f"cache_enabled={self._enable_cache})"
        )
