"""Geography — the spatial structure of the world, loaded on demand from real data."""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from db.models import Location as LocationModel
from db.models import LocationConnection as ConnectionModel

logger = logging.getLogger(__name__)

MAX_CONNECTION_DISTANCE_KM = 10.0


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
    admin_level_1: str | None  # Country
    admin_level_2: str | None  # Nation / Region
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


def _location_from_model(m: LocationModel) -> LocationData:
    return LocationData(
        id=m.id, name=m.name, type=m.type,
        lat=m.lat, lng=m.lng, elevation=m.elevation,
        terrain=m.terrain,
        admin_level_1=m.admin_level_1, admin_level_2=m.admin_level_2,
        admin_level_3=m.admin_level_3, admin_level_4=m.admin_level_4,
        population=m.population, metadata=m.metadata_,
    )


class _LRUCache:
    """Simple LRU cache backed by OrderedDict."""

    def __init__(self, max_size: int):
        self._data: OrderedDict = OrderedDict()
        self._max_size = max_size

    def get(self, key):
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._max_size:
            self._data.popitem(last=False)

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self._data)

    def clear(self):
        self._data.clear()


class Geography:
    """
    The spatial structure of the world — on demand.

    Nothing is loaded into memory at startup. Locations and connections
    are fetched from the database when needed and cached with LRU eviction.
    Same LOD pattern as Earth-proxy resolution: the world resolves detail
    as agents approach, not by cramming everything in at once.
    """

    def __init__(self, session_factory: async_sessionmaker):
        self._session_factory = session_factory
        self._loc_cache = _LRUCache(max_size=20_000)
        self._conn_cache = _LRUCache(max_size=10_000)
        self._location_count: int | None = None
        self._connection_count: int | None = None

    @property
    def location_count(self) -> int:
        return self._location_count or 0

    @property
    def connection_count(self) -> int:
        return self._connection_count or 0

    # --- Sync reads (cache only, used by agent tick) ---

    def get_location(self, location_id: int) -> LocationData | None:
        return self._loc_cache.get(location_id)

    def get_neighbours(self, location_id: int) -> list[ConnectionData]:
        result = self._conn_cache.get(location_id)
        return result if result is not None else []

    def get_nearby_locations(self, location_id: int) -> list[tuple[LocationData, ConnectionData]]:
        result = []
        for conn in self.get_neighbours(location_id):
            other_id = conn.to_id if conn.from_id == location_id else conn.from_id
            other_loc = self.get_location(other_id)
            if other_loc:
                result.append((other_loc, conn))
        return sorted(result, key=lambda x: x[1].distance_km)

    # --- Async reads (query DB on cache miss) ---

    async def get_location_async(self, location_id: int) -> LocationData | None:
        cached = self._loc_cache.get(location_id)
        if cached is not None:
            return cached
        async with self._session_factory() as session:
            result = await session.execute(
                select(LocationModel).where(LocationModel.id == location_id)
            )
            model = result.scalar_one_or_none()
            if model is None:
                return None
            loc = _location_from_model(model)
            self._loc_cache.put(location_id, loc)
            return loc

    async def get_neighbours_async(self, location_id: int) -> list[ConnectionData]:
        cached = self._conn_cache.get(location_id)
        if cached is not None:
            return cached
        await self.warm_connections({location_id})
        return self._conn_cache.get(location_id) or []

    async def get_nearby_locations_async(self, location_id: int) -> list[tuple[LocationData, ConnectionData]]:
        result = []
        for conn in await self.get_neighbours_async(location_id):
            other_id = conn.to_id if conn.from_id == location_id else conn.from_id
            other_loc = await self.get_location_async(other_id)
            if other_loc:
                result.append((other_loc, conn))
        return sorted(result, key=lambda x: x[1].distance_km)

    async def get_location_by_name_async(self, name: str) -> LocationData | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(LocationModel).where(func.lower(LocationModel.name) == name.lower()).limit(1)
            )
            model = result.scalar_one_or_none()
            if model is None:
                return None
            loc = _location_from_model(model)
            self._loc_cache.put(loc.id, loc)
            return loc

    # --- Bulk DB queries (for API, agent bootstrap, wind stations) ---

    async def query_locations(
        self,
        type: str | None = None,
        region: str | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LocationData]:
        """Query locations from DB with filters. For API endpoints."""
        async with self._session_factory() as session:
            q = select(LocationModel)
            if type:
                q = q.where(LocationModel.type == type)
            if region:
                q = q.where(func.lower(LocationModel.admin_level_2) == region.lower())
            if search:
                q = q.where(LocationModel.name.ilike(f"%{search}%"))
            q = q.order_by(LocationModel.population.desc().nullslast()).offset(offset).limit(limit)
            result = await session.execute(q)
            locations = []
            for model in result.scalars().all():
                loc = _location_from_model(model)
                self._loc_cache.put(loc.id, loc)
                locations.append(loc)
            return locations

    # Zoom -> additional location types to include. Lower zoom = sparser
    # set (overview); higher zoom = denser set (close-in detail).
    # All tiers always include capital + city.
    _GEOJSON_ZOOM_TYPES = [
        # (min_zoom_for_this_set, types_added)
        (0,  ("capital", "city")),
        (5,  ("town",)),
        (9,  ("village",)),
        (12, ("settlement",)),
    ]

    @classmethod
    def _types_for_zoom(cls, zoom: float | None) -> tuple[str, ...]:
        """Return the location types to include for a given zoom level."""
        z = 0 if zoom is None else float(zoom)
        included: list[str] = []
        for min_z, types in cls._GEOJSON_ZOOM_TYPES:
            if z >= min_z:
                included.extend(types)
        return tuple(included)

    async def query_locations_geojson(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        zoom: float | None = None,
    ) -> list[dict]:
        """Populated locations as GeoJSON features for map rendering.

        Args:
          bbox: optional (west, south, east, north) viewport. When provided,
                only locations inside the box are returned.
          zoom: optional MapLibre zoom level. Determines which place types
                are included — lower zoom = sparser set (capitals/cities only),
                higher zoom = adds town, village, settlement. See
                `_GEOJSON_ZOOM_TYPES`.

        Volume is managed at the right layer (viewport + zoom from the client),
        not by a hardcoded population threshold. Default (no params) returns
        the global overview tier (capital + city) — small enough for a fast
        initial load.
        """
        types = self._types_for_zoom(zoom)
        stmt = select(
            LocationModel.id, LocationModel.name, LocationModel.type,
            LocationModel.lat, LocationModel.lng,
            LocationModel.population, LocationModel.admin_level_2,
        ).where(LocationModel.type.in_(types))

        if bbox is not None:
            west, south, east, north = bbox
            # Simple lat/lng bounding box. Doesn't handle the antimeridian,
            # which is fine for typical viewports; if the box wraps the
            # 180°E/W line MapLibre splits it into two requests itself.
            stmt = stmt.where(
                LocationModel.lat >= south,
                LocationModel.lat <= north,
                LocationModel.lng >= west,
                LocationModel.lng <= east,
            )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [row.lng, row.lat]},
                    "properties": {
                        "id": row.id, "name": row.name, "type": row.type,
                        "population": row.population or 0,
                        "admin_level_2": row.admin_level_2 or "",
                    },
                }
                for row in result.all()
            ]

    async def get_top_locations_by_population(self, types: list[str], limit: int = 50) -> list[LocationData]:
        """Get top populated locations of given types. For weather/wind station selection."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(LocationModel)
                .where(LocationModel.type.in_(types))
                .where(LocationModel.population > 0)
                .order_by(LocationModel.population.desc())
                .limit(limit)
            )
            locations = []
            for model in result.scalars().all():
                loc = _location_from_model(model)
                self._loc_cache.put(loc.id, loc)
                locations.append(loc)
            return locations

    async def get_random_location_ids(self, count: int = 10) -> list[int]:
        """Get random location IDs from populated places. For agent broad exploration."""
        async with self._session_factory() as session:
            result = await session.execute(
                text(f"""
                    SELECT id FROM locations
                    WHERE type IN ('capital', 'city', 'town', 'village')
                    ORDER BY random() LIMIT {count}
                """)
            )
            return [row[0] for row in result.all()]

    async def get_spawn_locations(self, types: list[str], limit: int = 200) -> list[LocationData]:
        """Get candidate spawn locations for agents, sorted by population."""
        return await self.get_top_locations_by_population(types, limit)

    # --- Cache warming (LOD pattern — called before agent tick) ---

    async def warm(self, location_ids: set[int]) -> None:
        """Batch-fetch locations and connections for a set of IDs."""
        await self._warm_locations(location_ids)
        await self.warm_connections(location_ids)

    # asyncpg's `bind_execute` is limited to 32767 parameters per statement
    # (Postgres wire protocol uses int16 for the parameter count). With many
    # agents (>1000) the neighbour-ID set easily exceeds that, so all bulk
    # IN queries are chunked.
    _IN_CHUNK_SIZE = 5000

    @staticmethod
    def _chunked(items, n):
        items = list(items)
        for i in range(0, len(items), n):
            yield items[i:i + n]

    async def _warm_locations(self, location_ids: set[int]) -> None:
        missing = [lid for lid in location_ids if lid not in self._loc_cache]
        if not missing:
            return
        async with self._session_factory() as session:
            for chunk in self._chunked(missing, self._IN_CHUNK_SIZE):
                result = await session.execute(
                    select(LocationModel).where(LocationModel.id.in_(chunk))
                )
                for model in result.scalars().all():
                    loc = _location_from_model(model)
                    self._loc_cache.put(loc.id, loc)

    async def warm_connections(self, location_ids: set[int]) -> None:
        """Batch-fetch connections for multiple locations in one query."""
        missing = [lid for lid in location_ids if lid not in self._conn_cache]
        if not missing:
            return

        # Pre-populate with empty lists
        for lid in missing:
            self._conn_cache.put(lid, [])

        # The OR of two `IN(...)` clauses doubles the parameter count, so
        # halve the chunk size to stay safely below 32767.
        chunk_size = self._IN_CHUNK_SIZE // 2

        neighbour_ids: set[int] = set()
        async with self._session_factory() as session:
            for chunk in self._chunked(missing, chunk_size):
                result = await session.execute(
                    select(ConnectionModel).where(
                        (ConnectionModel.from_id.in_(chunk) | ConnectionModel.to_id.in_(chunk))
                        & (ConnectionModel.distance_km <= MAX_CONNECTION_DISTANCE_KM)
                    )
                )
                for row in result.scalars().all():
                    conn = ConnectionData(
                        from_id=row.from_id, to_id=row.to_id,
                        distance_km=row.distance_km,
                        connection_type=row.connection_type,
                        route_name=row.route_name, direction=row.direction,
                    )
                    if row.from_id in self._conn_cache:
                        self._conn_cache.get(row.from_id).append(conn)
                    if row.to_id in self._conn_cache:
                        self._conn_cache.get(row.to_id).append(conn)
                    neighbour_ids.add(row.from_id)
                    neighbour_ids.add(row.to_id)

        # Also warm the location data for neighbours so get_nearby_locations works.
        # `_warm_locations` is itself chunked.
        await self._warm_locations(neighbour_ids)

    # --- Counts ---

    async def load_counts(self) -> None:
        async with self._session_factory() as session:
            loc_result = await session.execute(text("SELECT COUNT(*) FROM locations"))
            self._location_count = loc_result.scalar_one()
            conn_result = await session.execute(
                text(f"SELECT COUNT(*) FROM location_connections WHERE distance_km <= {MAX_CONNECTION_DISTANCE_KM}")
            )
            self._connection_count = conn_result.scalar_one()

    def clear_caches(self) -> None:
        self._loc_cache.clear()
        self._conn_cache.clear()

    # --- Compat properties for code that checks geography.locations ---

    @property
    def locations(self) -> dict:
        """Access the location cache dict directly. For backwards compat only."""
        return self._loc_cache._data


async def load_geography(session_factory: async_sessionmaker) -> Geography:
    """Create geography with on-demand loading. No bulk data loaded at startup."""
    geo = Geography(session_factory=session_factory)
    await geo.load_counts()
    logger.info(
        f"Geography loaded: {geo.location_count} locations, "
        f"{geo.connection_count} connections available "
        f"(<= {MAX_CONNECTION_DISTANCE_KM}km, on-demand via cache)"
    )
    return geo
