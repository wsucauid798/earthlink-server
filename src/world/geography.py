"""Geography — the spatial structure of the world, loaded from real data."""

import logging
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from db.models import Location as LocationModel
from db.models import LocationConnection as ConnectionModel

logger = logging.getLogger(__name__)


@dataclass
class LocationData:
    """A real place in the world."""
    id: int
    name: str
    type: str
    lat: float
    lng: float
    elevation: float | None
    terrain: str | None
    admin_level_1: str | None  # United Kingdom
    admin_level_2: str | None  # England / Scotland / Wales / Northern Ireland
    admin_level_3: str | None  # County / Region
    admin_level_4: str | None  # District
    population: int | None
    metadata: dict | None


@dataclass
class ConnectionData:
    """A real connection between two places."""
    from_id: int
    to_id: int
    distance_km: float
    connection_type: str
    route_name: str | None
    direction: str | None


@dataclass
class Geography:
    """
    The full spatial structure of the world.
    Loaded from the database — these are real places and real connections.
    """
    locations: dict[int, LocationData] = field(default_factory=dict)
    connections: list[ConnectionData] = field(default_factory=list)
    _adjacency: dict[int, list[ConnectionData]] = field(default_factory=dict)

    @property
    def location_count(self) -> int:
        return len(self.locations)

    @property
    def connection_count(self) -> int:
        return len(self.connections)

    def get_location(self, location_id: int) -> LocationData | None:
        return self.locations.get(location_id)

    def get_location_by_name(self, name: str) -> LocationData | None:
        """Find a location by name (case-insensitive, returns first match)."""
        name_lower = name.lower()
        for loc in self.locations.values():
            if loc.name.lower() == name_lower:
                return loc
        return None

    def get_neighbours(self, location_id: int) -> list[ConnectionData]:
        """Get all connections from a location."""
        return self._adjacency.get(location_id, [])

    def get_nearby_locations(self, location_id: int) -> list[tuple[LocationData, ConnectionData]]:
        """Get neighbouring locations with their connection info."""
        result = []
        for conn in self.get_neighbours(location_id):
            other_id = conn.to_id if conn.from_id == location_id else conn.from_id
            other_loc = self.locations.get(other_id)
            if other_loc:
                result.append((other_loc, conn))
        return sorted(result, key=lambda x: x[1].distance_km)

    def get_locations_by_type(self, loc_type: str) -> list[LocationData]:
        """Get all locations of a given type."""
        return [loc for loc in self.locations.values() if loc.type == loc_type]

    def get_locations_in_region(self, admin_level_2: str) -> list[LocationData]:
        """Get all locations within a nation (England, Scotland, etc.)."""
        return [loc for loc in self.locations.values() if loc.admin_level_2 == admin_level_2]


async def load_geography(session: AsyncSession) -> Geography:
    """Load the full geography from the database."""
    geo = Geography()

    # Load all locations
    result = await session.execute(select(LocationModel))
    for loc_model in result.scalars().all():
        geo.locations[loc_model.id] = LocationData(
            id=loc_model.id,
            name=loc_model.name,
            type=loc_model.type,
            lat=loc_model.lat,
            lng=loc_model.lng,
            elevation=loc_model.elevation,
            terrain=loc_model.terrain,
            admin_level_1=loc_model.admin_level_1,
            admin_level_2=loc_model.admin_level_2,
            admin_level_3=loc_model.admin_level_3,
            admin_level_4=loc_model.admin_level_4,
            population=loc_model.population,
            metadata=loc_model.metadata_,
        )

    # Load connections (filtered by distance to reduce memory usage)
    # Only load connections under 10km to keep memory reasonable (~2.4M instead of 5M)
    # This still allows navigation while reducing memory by 52%
    MAX_CONNECTION_DISTANCE_KM = 10.0
    result = await session.execute(
        select(ConnectionModel).where(ConnectionModel.distance_km <= MAX_CONNECTION_DISTANCE_KM)
    )
    loaded_count = 0
    for conn_model in result.scalars().all():
        conn = ConnectionData(
            from_id=conn_model.from_id,
            to_id=conn_model.to_id,
            distance_km=conn_model.distance_km,
            connection_type=conn_model.connection_type,
            route_name=conn_model.route_name,
            direction=conn_model.direction,
        )
        geo.connections.append(conn)
        loaded_count += 1

        # Build adjacency (bidirectional)
        if conn.from_id not in geo._adjacency:
            geo._adjacency[conn.from_id] = []
        geo._adjacency[conn.from_id].append(conn)

        if conn.to_id not in geo._adjacency:
            geo._adjacency[conn.to_id] = []
        geo._adjacency[conn.to_id].append(conn)

    logger.info(
        f"Geography loaded: {geo.location_count} locations, {loaded_count} connections "
        f"(<= {MAX_CONNECTION_DISTANCE_KM}km filter applied for memory efficiency)"
    )
    return geo
