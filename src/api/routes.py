"""REST API routes for the EarthLink world server."""

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    AgentAnswerSchema,
    AgentDetailSchema,
    AgentSummarySchema,
    AstronomySchema,
    LocationSchema,
    NearbyLocationSchema,
    ConnectionSchema,
    SimulationConfigSchema,
    SimulationControlSchema,
    WeatherSchema,
    WorldStateSchema,
)
from world import World

router = APIRouter()

# The world instance — set during app startup
_world: World | None = None


def set_world(world: World) -> None:
    global _world
    _world = world


def get_world() -> World:
    if _world is None:
        raise HTTPException(status_code=503, detail="World not loaded")
    return _world


# --- Version ---

@router.get("/version")
async def get_version():
    """Return the server version."""
    from world import __version__
    return {"version": __version__}


# --- World State ---

@router.get("/world/state", response_model=WorldStateSchema)
async def get_world_state():
    """Get the current state of the world."""
    world = get_world()
    return world.get_state_summary()


# --- Agents ---

@router.get("/agents", response_model=list[AgentSummarySchema])
async def list_agents(
    search: str | None = Query(None, description="Search by agent name or ID (case-insensitive partial match)"),
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """List autonomous agents with current state and learning progress."""
    world = get_world()
    agents = world.list_agents()

    if search:
        search_lower = search.lower()
        agents = [a for a in agents if search_lower in a["name"].lower() or search_lower in a["id"].lower()]

    total = len(agents)
    agents = agents[offset : offset + limit]
    return agents


@router.get("/agents/{agent_id}", response_model=AgentDetailSchema)
async def get_agent(agent_id: str):
    """Get detailed state and learned knowledge for an autonomous agent."""
    world = get_world()
    agent = world.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.get("/agents/{agent_id}/ask", response_model=AgentAnswerSchema)
async def ask_agent(agent_id: str, question: str = Query(..., min_length=1, description="Question to ask the agent")):
    """Ask an agent a direct question and receive a memory-grounded answer."""
    world = get_world()
    answer = world.ask_agent(agent_id, question)
    if not answer:
        raise HTTPException(status_code=404, detail="Agent not found")
    return answer


# --- Locations ---
# Define /locations/geojson BEFORE /locations/{location_id} so "geojson" is not parsed as location_id (422).

@router.get("/locations/geojson")
async def get_locations_geojson():
    """All locations as a GeoJSON FeatureCollection — for map rendering."""
    world = get_world()
    features = []
    for loc in world.geography.locations.values():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [loc.lng, loc.lat],
            },
            "properties": {
                "id": loc.id,
                "name": loc.name,
                "type": loc.type,
                "population": loc.population or 0,
                "admin_level_2": loc.admin_level_2 or "",
            },
        })
    return {"type": "FeatureCollection", "features": features}


@router.get("/locations", response_model=list[LocationSchema])
async def list_locations(
    type: str | None = Query(None, description="Filter by location type (city, town, village, etc.)"),
    region: str | None = Query(None, description="Filter by nation (England, Scotland, Wales, Northern Ireland)"),
    search: str | None = Query(None, description="Search by name (case-insensitive partial match)"),
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """List locations in the world."""
    world = get_world()
    locations = list(world.geography.locations.values())

    if type:
        locations = [loc for loc in locations if loc.type == type]
    if region:
        locations = [loc for loc in locations if loc.admin_level_2 and loc.admin_level_2.lower() == region.lower()]
    if search:
        search_lower = search.lower()
        locations = [loc for loc in locations if search_lower in loc.name.lower()]

    total = len(locations)
    locations = locations[offset : offset + limit]

    return [
        LocationSchema(
            id=loc.id, name=loc.name, type=loc.type, lat=loc.lat, lng=loc.lng,
            elevation=loc.elevation, terrain=loc.terrain,
            admin_level_1=loc.admin_level_1, admin_level_2=loc.admin_level_2,
            admin_level_3=loc.admin_level_3, admin_level_4=loc.admin_level_4,
            population=loc.population,
        )
        for loc in locations
    ]


@router.get("/locations/{location_id}", response_model=LocationSchema)
async def get_location(location_id: int):
    """Get a specific location by ID."""
    world = get_world()
    loc = world.geography.get_location(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")
    return LocationSchema(
        id=loc.id, name=loc.name, type=loc.type, lat=loc.lat, lng=loc.lng,
        elevation=loc.elevation, terrain=loc.terrain,
        admin_level_1=loc.admin_level_1, admin_level_2=loc.admin_level_2,
        admin_level_3=loc.admin_level_3, admin_level_4=loc.admin_level_4,
        population=loc.population,
    )


@router.get("/locations/{location_id}/nearby", response_model=list[NearbyLocationSchema])
async def get_nearby_locations(location_id: int):
    """Get locations connected to a specific location."""
    world = get_world()
    loc = world.geography.get_location(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")

    nearby = world.geography.get_nearby_locations(location_id)
    return [
        NearbyLocationSchema(
            location=LocationSchema(
                id=n_loc.id, name=n_loc.name, type=n_loc.type, lat=n_loc.lat, lng=n_loc.lng,
                elevation=n_loc.elevation, terrain=n_loc.terrain,
                admin_level_1=n_loc.admin_level_1, admin_level_2=n_loc.admin_level_2,
                admin_level_3=n_loc.admin_level_3, admin_level_4=n_loc.admin_level_4,
                population=n_loc.population,
            ),
            connection=ConnectionSchema(
                from_id=conn.from_id, to_id=conn.to_id,
                distance_km=conn.distance_km, connection_type=conn.connection_type,
                route_name=conn.route_name, direction=conn.direction,
            ),
        )
        for n_loc, conn in nearby
    ]


# --- Weather ---

@router.get("/weather/{location_id}", response_model=WeatherSchema | None)
async def get_weather(location_id: int):
    """Get current weather at a location."""
    world = get_world()
    loc = world.geography.get_location(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")

    ws = world.weather.get_weather(location_id)
    if not ws:
        return None

    return WeatherSchema(
        location_id=ws.location_id,
        temperature_c=round(ws.temperature_c, 1) if ws.temperature_c else None,
        precipitation_mm=round(ws.precipitation_mm, 2) if ws.precipitation_mm else None,
        humidity_pct=round(ws.humidity_pct, 1) if ws.humidity_pct else None,
        wind_speed_kmh=round(ws.wind_speed_kmh, 1) if ws.wind_speed_kmh else None,
        wind_direction_deg=round(ws.wind_direction_deg, 1) if ws.wind_direction_deg else None,
        cloud_cover_pct=round(ws.cloud_cover_pct, 1) if ws.cloud_cover_pct else None,
        visibility_km=round(ws.visibility_km, 1) if ws.visibility_km else None,
        pressure_hpa=round(ws.pressure_hpa, 1) if ws.pressure_hpa else None,
        conditions=ws.conditions,
    )


# --- Astronomy ---

@router.get("/astronomy/{location_id}", response_model=AstronomySchema | None)
async def get_astronomy(location_id: int):
    """Get current astronomical state at a location (sunrise, sunset, daylight, moon)."""
    world = get_world()
    astro = world.time.get_astronomy(location_id)
    if not astro:
        return None

    return AstronomySchema(
        sunrise=astro.sunrise.isoformat() if astro.sunrise else None,
        sunset=astro.sunset.isoformat() if astro.sunset else None,
        day_length_hours=astro.day_length_hours,
        is_daylight=astro.is_daylight,
    )


# --- Time ---

@router.get("/time")
async def get_time():
    """Get the current world time."""
    world = get_world()
    return world.time.to_dict()


# --- Simulation Control ---

@router.post("/simulation/control")
async def control_simulation(control: SimulationControlSchema):
    """Start, pause, or reset the world."""
    world = get_world()

    if control.action == "start":
        await world.start()
        return {"status": "running"}
    elif control.action == "pause":
        await world.pause()
        return {"status": "paused"}
    elif control.action == "reset":
        await world.reset()
        return {"status": "reset"}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {control.action}")


@router.post("/simulation/config")
async def configure_simulation(config: SimulationConfigSchema):
    """Update simulation parameters (tick interval)."""
    world = get_world()

    if config.tick_interval_seconds is not None:
        world.config.tick_interval_seconds = config.tick_interval_seconds

    return {
        "status": "configured",
        "config": world.config.model_dump(),
        "time": world.time.to_dict(),
    }
