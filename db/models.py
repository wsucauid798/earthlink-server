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
    sunrise = Column(DateTime, nullable=True)
    sunset = Column(DateTime, nullable=True)
    day_length_hours = Column(Float, nullable=True)
    moon_phase = Column(Float, nullable=True)  # 0.0 = new moon, 0.5 = full moon, 1.0 = new moon

    # Relationships
    location = relationship("Location", back_populates="astronomy_data")


class WorldState(Base):
    __tablename__ = "world_state"

    id = Column(Integer, primary_key=True, default=1)
    current_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    tick_count = Column(Integer, nullable=False, default=0)
    is_running = Column(Boolean, nullable=False, default=False)
