"""Pydantic schemas for API request/response models."""

from datetime import datetime

from pydantic import BaseModel


# --- Location ---

class LocationSchema(BaseModel):
    id: int
    name: str
    type: str
    lat: float
    lng: float
    elevation: float | None = None
    terrain: str | None = None
    admin_level_1: str | None = None
    admin_level_2: str | None = None
    admin_level_3: str | None = None
    admin_level_4: str | None = None
    population: int | None = None

    class Config:
        from_attributes = True


class ConnectionSchema(BaseModel):
    from_id: int
    to_id: int
    distance_km: float
    connection_type: str
    route_name: str | None = None
    direction: str | None = None


class NearbyLocationSchema(BaseModel):
    location: LocationSchema
    connection: ConnectionSchema


# --- Weather ---

class WeatherSchema(BaseModel):
    location_id: int
    temperature_c: float | None = None
    precipitation_mm: float | None = None
    humidity_pct: float | None = None
    wind_speed_kmh: float | None = None
    wind_direction_deg: float | None = None
    cloud_cover_pct: float | None = None
    visibility_km: float | None = None
    pressure_hpa: float | None = None
    conditions: str | None = None


# --- Wind ---

class WindSchema(BaseModel):
    location_id: int
    speed_kmh: float | None = None
    direction_deg: float | None = None
    gust_kmh: float | None = None
    speed_upper_kmh: float | None = None
    direction_upper_deg: float | None = None
    upper_height_m: int | None = None
    pressure_hpa: float | None = None
    beaufort: int = 0
    beaufort_description: str = "Calm"
    terrain_modifier: str | None = None
    is_interpolated: bool = False
    interpolation_distance_km: float | None = None


# --- Time ---

class TimeSchema(BaseModel):
    current_time: str  # UTC ISO-8601
    local_time: str | None = None  # Local timezone ISO-8601
    tick_count: int
    date: str
    hour: int
    minute: int
    season: str
    timezone: str = "Europe/London"
    timezone_abbr: str = "GMT"  # e.g. "GMT" or "BST"
    utc_offset: str = "+00:00"  # e.g. "+00:00" or "+01:00"


# --- Astronomy ---

class AstronomySchema(BaseModel):
    # Sun
    sunrise: str | None = None
    sunset: str | None = None
    solar_noon: str | None = None
    day_length_hours: float | None = None
    is_daylight: bool = True
    solar_elevation_deg: float | None = None
    solar_azimuth_deg: float | None = None

    # Twilight
    civil_dawn: str | None = None
    civil_dusk: str | None = None
    nautical_dawn: str | None = None
    nautical_dusk: str | None = None

    # Moon
    moon_phase: float | None = None
    moon_phase_name: str | None = None
    moon_phase_emoji: str | None = None
    moon_illumination_pct: float | None = None
    moon_age_days: float | None = None
    moonrise: str | None = None
    moonset: str | None = None


# --- Geophysics ---

class GeophysicsSchema(BaseModel):
    gravity_ms2: float
    gravity_base_ms2: float
    tidal_variation_ms2: float
    magnetic_field_ut: float
    magnetic_declination_deg: float
    magnetic_inclination_deg: float
    rotation_speed_kmh: float


# --- Atmosphere ---

class AtmosphereSchema(BaseModel):
    # Air quality (fetched)
    european_aqi: int | None = None
    european_aqi_label: str | None = None
    us_aqi: int | None = None
    pm2_5: float | None = None
    pm10: float | None = None
    ozone: float | None = None
    nitrogen_dioxide: float | None = None
    sulphur_dioxide: float | None = None
    carbon_monoxide: float | None = None
    # Derived (computed)
    dew_point_c: float | None = None
    feels_like_c: float | None = None
    uv_index: float | None = None
    air_density_kgm3: float | None = None


# --- World State ---

class GeographyStatsSchema(BaseModel):
    location_types: dict[str, int] = {}
    countries: dict[str, int] = {}
    regions: dict[str, int] = {}
    total_population: int = 0
    elevation_min: float | None = None
    elevation_max: float | None = None


class WorldStateSchema(BaseModel):
    time: TimeSchema | None = None
    is_running: bool = False
    location_count: int = 0
    connection_count: int = 0
    weather_stations: int = 0
    wind_stations: int = 0
    weather: dict[str, dict] = {}
    geography_stats: GeographyStatsSchema | None = None
    agent_count: int = 0
    agent_backend: str = ""
    agents: list[dict] = []
    earth_proxy: dict | None = None
    refresh: dict[str, dict] | None = None


# --- Agents ---

class AgentTopLocationSchema(BaseModel):
    location_id: int
    location_name: str | None = None
    score: float
    visits: int


class AgentVisitedPlaceSchema(BaseModel):
    location_id: int
    location_name: str | None = None
    visits: int


class AgentSummarySchema(BaseModel):
    id: str
    name: str
    location_id: int
    location_name: str | None = None
    last_action: str
    energy: float
    knowledge_score: float
    visited_locations: int
    policy: str
    last_reward: float
    goal: dict | None = None


class AgentDetailSchema(AgentSummarySchema):
    top_locations: list[AgentTopLocationSchema] = []
    known_conditions: dict[str, int] = {}
    visited_places: list[AgentVisitedPlaceSchema] = []


class AgentAnswerSchema(BaseModel):
    agent_id: str
    question: str
    answer: str
    visited_places: list[AgentVisitedPlaceSchema] = []
    retrieval_backend: str
    answer_confidence: float
    answer_certainty: str
    supporting_facts: list[dict] = []


# --- Simulation Control ---

class SimulationControlSchema(BaseModel):
    action: str  # "start", "pause", "reset"


class SimulationConfigSchema(BaseModel):
    tick_interval_seconds: float | None = None  # Real seconds between world ticks


# --- Tick Event (WebSocket) ---

class TickEventSchema(BaseModel):
    tick: int
    time: TimeSchema
    weather_updated: bool
    wind_updated: bool
    atmosphere_updated: bool
    astronomy_updated: bool
    data_feeds_updated: bool
    agent_events: list[dict] = []


# --- Orbital ---

class OrbitalSchema(BaseModel):
    earth_sun_distance_km: float
    earth_sun_distance_au: float
    orbital_position_deg: float
    true_anomaly_deg: float
    mean_anomaly_deg: float
    orbital_speed_kms: float
    axial_tilt_deg: float
    solar_declination_deg: float
    season: str
    season_progress: float
    days_to_perihelion: float
    days_to_next_event: float
    next_event: str
    eccentricity: float
    semi_major_axis_km: float


# --- Data Feeds ---

class SolarActivitySchema(BaseModel):
    kp_index: float | None = None
    kp_category: str | None = None
    solar_wind_speed_kms: float | None = None
    solar_wind_density: float | None = None
    solar_wind_temperature_k: float | None = None
    bz_gsm_nt: float | None = None
    bt_nt: float | None = None
    xray_flux: float | None = None
    xray_class: str | None = None
    last_updated: str | None = None
