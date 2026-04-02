"""World configuration — settings that define how the world behaves."""

from enum import Enum
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
    wind: RefreshPolicy = RefreshPolicy(enabled=True, interval_minutes=30)
    astronomy: RefreshPolicy = RefreshPolicy(enabled=True, interval_minutes=1440)  # daily
    atmosphere: RefreshPolicy = RefreshPolicy(enabled=True, interval_minutes=30)
    geography: RefreshPolicy = RefreshPolicy(enabled=True, interval_minutes=10080)  # weekly
    data_feeds: RefreshPolicy = RefreshPolicy(enabled=True, interval_minutes=30)


class FallbackBehavior(str, Enum):
    """Fallback strategy when an Earth adapter fails or is rate-limited.

    Controls how EarthProxy handles adapter failures and rate limit exhaustion.
    """
    SKIP = "skip"              # Silent skip, return empty results
    USE_STALE = "use_stale"    # Return stale cache if available, else skip
    RETRY_AFTER = "retry_after"  # Wait and retry (for rate limits only)
    LOG_WARNING = "log_warning"  # Log warning, then skip (default)


class AdapterPolicy(BaseModel):
    """Configuration policy for a single Earth adapter.

    Similar to RefreshPolicy but for runtime adapter behavior control.
    Each adapter can be individually configured for rate limiting, caching,
    and fallback behavior.
    """
    enabled: bool = True

    # Rate limiting
    max_requests_per_second: float | None = None  # None = unlimited
    max_requests_per_minute: float | None = None  # None = unlimited

    # Cache control
    cache_ttl_seconds: int | None = None  # None = use EarthProxy default (600s)

    # Timeout — hard wall-clock cap on a single adapter.resolve() call
    timeout_seconds: float = 10.0  # Per-adapter override; 0 = use proxy default

    # Fallback behavior
    fallback: FallbackBehavior = FallbackBehavior.LOG_WARNING


class AdapterPolicies(BaseModel):
    """Policies for all Earth adapters.

    Provides a default policy that applies to all adapters, plus optional
    per-adapter overrides. Follows the same pattern as RefreshPolicies.

    Per-adapter fields correspond to adapter names (lowercase, underscored).
    If an adapter has no explicit override, it uses the default policy.
    """

    # Default policy applied to all adapters unless overridden
    default: AdapterPolicy = AdapterPolicy()

    # Per-adapter overrides (one field per adapter, all optional)
    # Encyclopedia / reference
    wikipedia: AdapterPolicy | None = None
    wikidata: AdapterPolicy | None = None
    dbpedia: AdapterPolicy | None = None

    # News / current events
    bbc_news: AdapterPolicy | None = None
    reuters: AdapterPolicy | None = None

    # Government / institutional
    legislation_gov_uk: AdapterPolicy | None = None
    ons: AdapterPolicy | None = None
    data_gov_uk: AdapterPolicy | None = None
    parliament: AdapterPolicy | None = None
    police_data: AdapterPolicy | None = None

    # Public discourse
    stack_exchange: AdapterPolicy | None = None
    mastodon: AdapterPolicy | None = None

    # Cultural / social
    bank_holidays: AdapterPolicy | None = None
    overpass: AdapterPolicy | None = None

    # Historical / archival
    internet_archive: AdapterPolicy | None = None
    gutenberg: AdapterPolicy | None = None
    british_museum: AdapterPolicy | None = None
    historic_england: AdapterPolicy | None = None
    national_archives: AdapterPolicy | None = None

    # Search / discovery
    duckduckgo: AdapterPolicy | None = None

    # Social / real-time discourse
    bluesky: AdapterPolicy | None = None

    # LLM substrate
    ollama: AdapterPolicy | None = None

    def get_policy(self, adapter_name: str) -> AdapterPolicy:
        """Get the effective policy for an adapter.

        Returns the adapter-specific override if one exists, otherwise
        returns the default policy.

        Args:
            adapter_name: Name of the adapter (e.g., "wikipedia", "bbc_news")

        Returns:
            AdapterPolicy: The effective policy for this adapter
        """
        # Convert adapter name to field name (e.g., "bbc_news" stays "bbc_news")
        field_name = adapter_name.lower().replace("-", "_")
        override = getattr(self, field_name, None)
        return override if override is not None else self.default


class WorldConfig(BaseModel):
    """Configuration for the virtual world.

    The world IS Earth. Time is always real Earth time — there is no
    simulated clock, no alternative era, no fast-forward. The world's
    clock is the Earth clock.
    """

    # Time
    tick_interval_seconds: float = 1.0  # Real seconds between ticks
    timezone: str = "Europe/London"  # IANA timezone — handles GMT/BST automatically

    # Tick wall-clock budget — tick() will time-cap the agent phase so the
    # total tick never exceeds this. WS broadcast happens regardless.
    max_tick_wall_seconds: float = 2.0

    # Region
    region: str = "GB"  # ISO country code for the region this world covers

    # Freshness — how often each data domain is refreshed from its source
    refresh: RefreshPolicies = RefreshPolicies()

    # Adapter policies — control over Earth adapter behavior (rate limits, caching, fallback)
    adapters: AdapterPolicies = AdapterPolicies()

    # Earth proxy concurrency — two independent limits
    adapter_timeout_seconds: float = 10.0   # Default hard timeout per adapter.resolve()
    max_concurrent_adapters: int = 8        # Adapters queried in parallel per location
    max_concurrent_locations: int = 4       # Locations resolved in parallel

    # Wind system
    wind_station_count: int = 100  # Number of wind monitoring stations

    # Autonomous agents
    agent_count: int = 300
    agent_learning_rate: float = 0.2
    agent_exploration_bias: float = 0.35
    agent_random_seed: int = 42
