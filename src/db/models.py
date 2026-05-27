from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship

# pgvector — vector column type for embedding storage / similarity search.
# Imported lazily-safe: the package is in requirements; if missing, the
# import will fail loudly at startup rather than silently falling back.
from pgvector.sqlalchemy import Vector

# Must match the dimension pinned in migration c5e7a09f8b21 and the
# embedding model selected in S71 (BAAI/bge-m3).
EMBEDDING_DIMS = 1024


class Base(DeclarativeBase):
    pass


class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    type = Column(String(100), nullable=False, index=True)  # city, town, village, river, mountain, etc.
    lat = Column(Float, nullable=False)
    lng = Column(Float, nullable=False)
    elevation = Column(Float, nullable=True)  # metres above sea level
    terrain = Column(String(100), nullable=True)  # urban, rural, coastal, highland, woodland, farmland
    admin_level_1 = Column(String(255), nullable=True)  # country (United Kingdom)
    admin_level_2 = Column(String(255), nullable=True)  # nation (England, Scotland, Wales, N. Ireland)
    admin_level_3 = Column(String(255), nullable=True)  # region/county
    admin_level_4 = Column(String(255), nullable=True)  # district/borough
    population = Column(BigInteger, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)  # additional real-world attributes

    # Relationships
    connections_from = relationship(
        "LocationConnection", foreign_keys="LocationConnection.from_id", back_populates="from_location"
    )
    connections_to = relationship(
        "LocationConnection", foreign_keys="LocationConnection.to_id", back_populates="to_location"
    )
    weather_data = relationship("WeatherData", back_populates="location")
    astronomy_data = relationship("AstronomyData", back_populates="location")
    wind_data = relationship("WindData", back_populates="location")


class LocationConnection(Base):
    __tablename__ = "location_connections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    from_id = Column(Integer, ForeignKey("locations.id"), nullable=False, index=True)
    to_id = Column(Integer, ForeignKey("locations.id"), nullable=False, index=True)
    distance_km = Column(Float, nullable=False)
    connection_type = Column(String(50), nullable=False)  # road, rail, path, waterway, proximity
    route_name = Column(String(255), nullable=True)  # e.g. "M1", "A1", "East Coast Main Line"
    direction = Column(String(20), nullable=True)  # N, NE, E, SE, S, SW, W, NW

    # Relationships
    from_location = relationship("Location", foreign_keys=[from_id], back_populates="connections_from")
    to_location = relationship("Location", foreign_keys=[to_id], back_populates="connections_to")


class WeatherData(Base):
    __tablename__ = "weather_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=False, index=True)
    datetime = Column(DateTime, nullable=False, index=True)
    temperature_c = Column(Float, nullable=True)
    precipitation_mm = Column(Float, nullable=True)
    humidity_pct = Column(Float, nullable=True)
    wind_speed_kmh = Column(Float, nullable=True)
    wind_direction_deg = Column(Float, nullable=True)
    cloud_cover_pct = Column(Float, nullable=True)
    visibility_km = Column(Float, nullable=True)
    pressure_hpa = Column(Float, nullable=True)
    conditions = Column(String(100), nullable=True)  # WMO weather code description

    # Relationships
    location = relationship("Location", back_populates="weather_data")


class AstronomyData(Base):
    __tablename__ = "astronomy_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    # Sun
    sunrise = Column(DateTime, nullable=True)
    sunset = Column(DateTime, nullable=True)
    solar_noon = Column(DateTime, nullable=True)
    day_length_hours = Column(Float, nullable=True)

    # Twilight
    civil_dawn = Column(DateTime, nullable=True)
    civil_dusk = Column(DateTime, nullable=True)
    nautical_dawn = Column(DateTime, nullable=True)
    nautical_dusk = Column(DateTime, nullable=True)

    # Moon
    moon_phase = Column(Float, nullable=True)              # 0.0–1.0
    moon_illumination_pct = Column(Float, nullable=True)   # 0–100
    moon_age_days = Column(Float, nullable=True)           # days since new moon
    moonrise = Column(DateTime, nullable=True)
    moonset = Column(DateTime, nullable=True)

    # Relationships
    location = relationship("Location", back_populates="astronomy_data")


class WindData(Base):
    __tablename__ = "wind_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=False, index=True)
    datetime = Column(DateTime, nullable=False, index=True)

    # Surface (10m)
    wind_speed_10m = Column(Float, nullable=True)       # km/h
    wind_direction_10m = Column(Float, nullable=True)    # degrees (0=N, 90=E, 180=S, 270=W)
    wind_gusts_10m = Column(Float, nullable=True)        # km/h

    # Upper level (80m for forecast, 100m for historical)
    wind_speed_upper = Column(Float, nullable=True)      # km/h
    wind_direction_upper = Column(Float, nullable=True)   # degrees
    upper_height_m = Column(Integer, nullable=True)       # 80 or 100

    # Atmospheric context
    pressure_hpa = Column(Float, nullable=True)

    # Relationships
    location = relationship("Location", back_populates="wind_data")


class WorldState(Base):
    __tablename__ = "world_state"

    id = Column(Integer, primary_key=True, default=1)
    current_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    tick_count = Column(Integer, nullable=False, default=0)
    is_running = Column(Boolean, nullable=False, default=False)


class AgentState(Base):
    __tablename__ = "agent_state"

    id = Column(String(32), primary_key=True)
    name = Column(String(255), nullable=False)
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=False, index=True)
    energy = Column(Float, nullable=False, default=100.0)
    last_move_distance_km = Column(Float, nullable=False, default=0.0)
    last_move_connection_type = Column(String(64), nullable=False, default="road")
    last_action = Column(String(255), nullable=False, default="spawned")

    learning_rate = Column(Float, nullable=False)
    exploration_bias = Column(Float, nullable=False)
    risk_tolerance = Column(Float, nullable=False)
    stamina = Column(Float, nullable=False)
    comfort_temperature_c = Column(Float, nullable=False)

    knowledge = Column(JSONB, nullable=False, default=dict)


class AgentFact(Base):
    """Per-agent semantic memory backed by pgvector.

    One row per fact the agent has learned. Embedding is computed at write
    time via the TEI service (S71: BAAI/bge-m3, 1024 dims). Read path uses
    cosine similarity on the HNSW index (see migration c5e7a09f8b21).

    Replaces the per-agent ChromaDB collections used by ChromaFactStore
    once the migration in S77 lands.
    """
    __tablename__ = "agent_facts"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    agent_id = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIMS), nullable=False)
    location_id = Column(Integer, ForeignKey("locations.id", ondelete="SET NULL"), nullable=True)
    tick_count = Column(Integer, nullable=True)
    fact_metadata = Column("metadata", JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
