"""World configuration — settings that define how the world behaves."""

from datetime import datetime, timezone

from pydantic import BaseModel


class WorldConfig(BaseModel):
    """Configuration for the virtual world."""

    # Time
    tick_interval_seconds: float = 1.0  # Real seconds between ticks
    time_scale_minutes: int = 15  # Simulated minutes per tick (1 tick = 15 min of world time)
    start_time: datetime = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # When the world begins

    # Region
    region: str = "GB"  # ISO country code for the region this world covers

    # Weather
    weather_update_interval_ticks: int = 4  # Update weather every N ticks (every simulated hour at default scale)
