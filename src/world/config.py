"""World configuration — settings that define how the world behaves."""

from pydantic import BaseModel


class RefreshPolicy(BaseModel):
    """Freshness policy for a world data domain.

    Every domain has a policy. The interval says how often to refresh.
    Set enabled=False to stop refreshing (data stays as loaded at startup).
    """
    enabled: bool = True
    interval_minutes: int  # Real-world minutes between refreshes


class RefreshPolicies(BaseModel):
    """Refresh policies for all world data domains.

    Each domain gets a policy. The world enforces them.
    """
    weather: RefreshPolicy = RefreshPolicy(enabled=True, interval_minutes=30)
    astronomy: RefreshPolicy = RefreshPolicy(enabled=True, interval_minutes=1440)  # daily
    geography: RefreshPolicy = RefreshPolicy(enabled=True, interval_minutes=10080)  # weekly


class WorldConfig(BaseModel):
    """Configuration for the virtual world.

    The world IS Earth. Time is always real Earth time — there is no
    simulated clock, no alternative era, no fast-forward. The world's
    clock is the UK clock.
    """

    # Time
    tick_interval_seconds: float = 1.0  # Real seconds between ticks
    timezone: str = "Europe/London"  # IANA timezone — handles GMT/BST automatically

    # Region
    region: str = "GB"  # ISO country code for the region this world covers

    # Freshness — how often each data domain is refreshed from its source
    refresh: RefreshPolicies = RefreshPolicies()

    # Autonomous agents
    agent_count: int = 3
    agent_learning_rate: float = 0.2
    agent_exploration_bias: float = 0.35
    agent_random_seed: int = 42
