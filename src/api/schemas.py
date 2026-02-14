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
    sunrise: str | None = None
    sunset: str | None = None
    day_length_hours: float | None = None
    is_daylight: bool = True


# --- World State ---

class WorldStateSchema(BaseModel):
    time: TimeSchema | None = None
    is_running: bool = False
    location_count: int = 0
    connection_count: int = 0
    weather_stations: int = 0
    weather: dict[str, dict] = {}
    agent_count: int = 0
    agents: list[dict] = []


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
    agent_events: list[dict] = []
