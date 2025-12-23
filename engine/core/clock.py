"""Clock abstraction for simulation engine.

Clock provides time advancement for the event loop.
Different implementations support historical and realtime modes.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Iterator


class Clock(ABC):
    """Abstract clock interface.

    The clock is responsible for:
    - Advancing time
    - Determining when simulation is done
    - Providing current time
    """

    @abstractmethod
    def advance(self) -> Optional[datetime]:
        """Advance to next timestamp.

        Returns:
            Next timestamp, or None if done
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """Check if simulation is complete.

        Returns:
            True if no more time to advance
        """
        pass

    @abstractmethod
    def current_time(self) -> datetime:
        """Get current simulation time.

        Returns:
            Current timestamp
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset clock to initial state."""
        pass
