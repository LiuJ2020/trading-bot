"""Historical clock for backtesting.

Iterates through historical timestamps at configurable speed.
"""

from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd

from engine.core.clock import Clock


class HistoricalClock(Clock):
    """Clock that iterates through historical timestamps.

    Used for backtesting. Can run at different speeds (1x, 10x, max).
    """

    def __init__(
        self,
        timestamps: List[datetime],
        speed: str = "max"
    ):
        """Initialize historical clock.

        Args:
            timestamps: Sorted list of timestamps to iterate through
            speed: Clock speed ("1x", "10x", "max")
                - "1x": Real-time speed (mainly for demo purposes)
                - "10x": 10x faster than real-time
                - "max": As fast as possible (typical for backtesting)
        """
        if not timestamps:
            raise ValueError("timestamps cannot be empty")

        # Sort timestamps to ensure proper ordering
        self._timestamps = sorted(timestamps)
        self._speed = speed
        self._index = -1  # Start before first timestamp
        self._current = None

    @classmethod
    def from_data(cls, data: pd.DataFrame, time_column: str = "timestamp", speed: str = "max") -> 'HistoricalClock':
        """Create clock from DataFrame.

        Args:
            data: DataFrame containing timestamps
            time_column: Name of timestamp column
            speed: Clock speed

        Returns:
            HistoricalClock instance
        """
        timestamps = pd.to_datetime(data[time_column]).unique().tolist()
        return cls(timestamps, speed)

    @classmethod
    def from_range(
        cls,
        start: datetime,
        end: datetime,
        freq: str = "1D",
        speed: str = "max"
    ) -> 'HistoricalClock':
        """Create clock from date range.

        Args:
            start: Start date
            end: End date
            freq: Frequency (pandas frequency string, e.g., "1D", "1H", "5T")
            speed: Clock speed

        Returns:
            HistoricalClock instance
        """
        timestamps = pd.date_range(start, end, freq=freq).tolist()
        return cls(timestamps, speed)

    def advance(self) -> Optional[datetime]:
        """Advance to next timestamp.

        Returns:
            Next timestamp, or None if done
        """
        self._index += 1
        if self._index >= len(self._timestamps):
            return None

        self._current = self._timestamps[self._index]

        # TODO: Implement speed control (sleep for 1x, 10x)
        # For now, just return immediately (max speed)

        return self._current

    def is_done(self) -> bool:
        """Check if all timestamps have been processed.

        Returns:
            True if simulation is complete
        """
        return self._index >= len(self._timestamps) - 1

    def current_time(self) -> datetime:
        """Get current timestamp.

        Returns:
            Current timestamp

        Raises:
            RuntimeError: If clock hasn't been advanced yet
        """
        if self._current is None:
            raise RuntimeError("Clock not started. Call advance() first.")
        return self._current

    def reset(self) -> None:
        """Reset clock to beginning."""
        self._index = -1
        self._current = None

    def peek_next(self) -> Optional[datetime]:
        """Peek at next timestamp without advancing.

        Returns:
            Next timestamp, or None if done
        """
        next_index = self._index + 1
        if next_index >= len(self._timestamps):
            return None
        return self._timestamps[next_index]

    def remaining_count(self) -> int:
        """Get number of remaining timestamps.

        Returns:
            Count of remaining timestamps
        """
        return max(0, len(self._timestamps) - self._index - 1)

    def progress_percentage(self) -> float:
        """Get progress percentage (0-100).

        Returns:
            Progress as percentage
        """
        if not self._timestamps:
            return 100.0
        return (self._index + 1) / len(self._timestamps) * 100

    def __len__(self) -> int:
        """Total number of timestamps."""
        return len(self._timestamps)

    def __repr__(self) -> str:
        status = f"{self._index + 1}/{len(self._timestamps)}" if self._current else "not started"
        return f"HistoricalClock(speed={self._speed}, progress={status})"
