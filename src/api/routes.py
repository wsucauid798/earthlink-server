"""REST API routes for the EarthLink world server."""

from fastapi import APIRouter, HTTPException, Query
from config import settings

from api.schemas import (
    AgentAnswerSchema,
    AgentDetailSchema,
    AgentSummarySchema,
    AtmosphereSchema,
    AstronomySchema,
    EvalSnapshotSchema,
    GeophysicsSchema,
    LocationSchema,
    NearbyLocationSchema,
    ConnectionSchema,
    OrbitalSchema,
    SimulationConfigSchema,
    SimulationControlSchema,
    SolarActivitySchema,
    WeatherSchema,
    WindSchema,
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
    wt_path = settings.wt_path.strip() or "/wt/world"
    if not wt_path.startswith("/"):
        wt_path = f"/{wt_path}"
    wt_url = settings.wt_public_url.strip() or None
    return {
        "version": __version__,
        "webtransport": {
            "enabled": settings.wt_enabled,
            "url": wt_url,
            "path": wt_path,
        },
    }


# --- World State ---

@router.get("/world/state", response_model=WorldStateSchema)
async def get_world_state():
    """Get the current state of the world."""
    world = get_world()
    return await world.get_state_summary()


# --- Agents ---

@router.get("/agents", response_model=list[AgentSummarySchema])
async def list_agents(
    search: str | None = Query(None, description="Search by agent name or ID (case-insensitive partial match)"),
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """List autonomous agents with current state and learning progress."""
    world = get_world()
    agents = await world.list_agents()

    if search:
        search_lower = search.lower()
        scored = []
        for a in agents:
            name_lower = a["name"].lower()
            if name_lower == search_lower:
                scored.append((0, a))
            elif name_lower.startswith(search_lower):
                scored.append((1, a))
            elif search_lower in name_lower:
                scored.append((2, a))
            elif search_lower in a["id"].lower():
                scored.append((3, a))
        scored.sort(key=lambda x: x[0])
        agents = [a for _, a in scored]

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
async def get_locations_geojson(
    bbox: str | None = Query(
        None,
        description="Optional viewport bbox 'west,south,east,north' in lng/lat. Returns only locations inside.",
    ),
    zoom: float | None = Query(
        None,
        ge=0,
        le=22,
        description="Optional MapLibre zoom. Lower zoom returns a sparser type tier (capital/city); higher zoom adds town, village, settlement.",
    ),
):
    """Populated locations as a GeoJSON FeatureCollection.

    Volume control is bbox + zoom from the client, not a hardcoded
    population filter. Default (no params) returns the global overview tier.
    """
    world = get_world()
    parsed_bbox: tuple[float, float, float, float] | None = None
    if bbox:
        try:
            parts = [float(p) for p in bbox.split(",")]
            if len(parts) != 4:
                raise ValueError
            parsed_bbox = (parts[0], parts[1], parts[2], parts[3])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="bbox must be 'west,south,east,north' as four floats",
            )
    features = await world.geography.query_locations_geojson(
        bbox=parsed_bbox, zoom=zoom,
    )
    return {"type": "FeatureCollection", "features": features}


@router.get("/locations", response_model=list[LocationSchema])
async def list_locations(
    type: str | None = Query(None, description="Filter by location type (city, town, village, etc.)"),
    region: str | None = Query(None, description="Filter by nation (England, Scotland, Wales, Northern Ireland)"),
    search: str | None = Query(None, description="Search by name (case-insensitive partial match)"),
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """List locations in the world — queried from DB."""
    world = get_world()
    locations = await world.geography.query_locations(
        type=type, region=region, search=search, limit=limit, offset=offset,
    )

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

    nearby = await world.geography.get_nearby_locations_async(location_id)
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


# --- Wind ---
# Define /wind/stations BEFORE /wind/{location_id} so "stations" is not parsed as location_id.

@router.get("/wind/stations", response_model=list[WindSchema])
async def get_wind_stations():
    """Get wind data for all direct monitoring stations."""
    world = get_world()
    if not world.wind:
        return []
    stations = world.wind.get_all_stations()
    return [
        WindSchema(
            location_id=ws.location_id,
            speed_kmh=round(ws.speed_kmh, 1) if ws.speed_kmh else None,
            direction_deg=round(ws.direction_deg, 1) if ws.direction_deg else None,
            gust_kmh=round(ws.gust_kmh, 1) if ws.gust_kmh else None,
            speed_upper_kmh=round(ws.speed_upper_kmh, 1) if ws.speed_upper_kmh else None,
            direction_upper_deg=round(ws.direction_upper_deg, 1) if ws.direction_upper_deg else None,
            upper_height_m=ws.upper_height_m,
            pressure_hpa=round(ws.pressure_hpa, 1) if ws.pressure_hpa else None,
            beaufort=ws.beaufort,
            beaufort_description=ws.beaufort_description,
            terrain_modifier=ws.terrain_modifier,
            is_interpolated=False,
        )
        for ws in stations.values()
    ]


@router.get("/wind/{location_id}", response_model=WindSchema | None)
async def get_wind(location_id: int):
    """Get current wind at any location (direct station or interpolated)."""
    world = get_world()
    loc = world.geography.get_location(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")

    if not world.wind:
        return None

    ws = world.wind.get_wind(
        location_id, lat=loc.lat, lng=loc.lng, terrain=loc.terrain,
    )
    if not ws:
        return None

    return WindSchema(
        location_id=ws.location_id,
        speed_kmh=round(ws.speed_kmh, 1) if ws.speed_kmh else None,
        direction_deg=round(ws.direction_deg, 1) if ws.direction_deg else None,
        gust_kmh=round(ws.gust_kmh, 1) if ws.gust_kmh else None,
        speed_upper_kmh=round(ws.speed_upper_kmh, 1) if ws.speed_upper_kmh else None,
        direction_upper_deg=round(ws.direction_upper_deg, 1) if ws.direction_upper_deg else None,
        upper_height_m=ws.upper_height_m,
        pressure_hpa=round(ws.pressure_hpa, 1) if ws.pressure_hpa else None,
        beaufort=ws.beaufort,
        beaufort_description=ws.beaufort_description,
        terrain_modifier=ws.terrain_modifier,
        is_interpolated=ws.is_interpolated,
        interpolation_distance_km=ws.interpolation_distance_km,
    )


# --- Astronomy ---

@router.get("/astronomy/{location_id}", response_model=AstronomySchema | None)
async def get_astronomy(location_id: int):
    """Get current astronomical state at a location (sun, moon, twilight)."""
    world = get_world()

    # Get location coordinates for real-time solar position
    lat, lng = None, None
    if world.geography:
        loc = world.geography.get_location(location_id)
        if loc:
            lat, lng = loc.lat, loc.lng

    astro = world.time.get_astronomy(location_id, lat=lat, lng=lng)
    if not astro:
        return None

    def _iso(dt):
        return dt.isoformat() if dt else None

    return AstronomySchema(
        # Sun
        sunrise=_iso(astro.sunrise),
        sunset=_iso(astro.sunset),
        solar_noon=_iso(astro.solar_noon),
        day_length_hours=astro.day_length_hours,
        is_daylight=astro.is_daylight,
        solar_elevation_deg=astro.solar_elevation_deg,
        solar_azimuth_deg=astro.solar_azimuth_deg,
        # Twilight
        civil_dawn=_iso(astro.civil_dawn),
        civil_dusk=_iso(astro.civil_dusk),
        nautical_dawn=_iso(astro.nautical_dawn),
        nautical_dusk=_iso(astro.nautical_dusk),
        # Moon
        moon_phase=astro.moon_phase,
        moon_phase_name=astro.moon_phase_name,
        moon_phase_emoji=astro.moon_phase_emoji,
        moon_illumination_pct=astro.moon_illumination_pct,
        moon_age_days=astro.moon_age_days,
        moonrise=_iso(astro.moonrise),
        moonset=_iso(astro.moonset),
    )


# --- Geophysics ---

@router.get("/geophysics/{location_id}", response_model=GeophysicsSchema)
async def get_geophysics(location_id: int):
    """Get geophysical data at a location (gravity, magnetic field, rotation)."""
    world = get_world()
    loc = world.geography.get_location(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")

    state = world.geophysics.compute(
        lat=loc.lat,
        lng=loc.lng,
        elevation=loc.elevation,
        dt=world.time.current_time.replace(tzinfo=None),
    )

    return GeophysicsSchema(
        gravity_ms2=state.gravity_ms2,
        gravity_base_ms2=state.gravity_base_ms2,
        tidal_variation_ms2=state.tidal_variation_ms2,
        magnetic_field_ut=state.magnetic_field_ut,
        magnetic_declination_deg=state.magnetic_declination_deg,
        magnetic_inclination_deg=state.magnetic_inclination_deg,
        rotation_speed_kmh=state.rotation_speed_kmh,
    )


# --- Atmosphere ---

@router.get("/atmosphere/{location_id}", response_model=AtmosphereSchema | None)
async def get_atmosphere(location_id: int):
    """Get atmospheric data at a location (air quality, dew point, UV, etc.)."""
    world = get_world()
    loc = world.geography.get_location(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")

    # Get current weather and astronomy for computed values
    weather = world.weather.get_weather(location_id) if world.weather else None
    astro = world.time.get_astronomy(location_id, lat=loc.lat, lng=loc.lng) if world.time else None

    state = world.atmosphere.get_atmosphere(
        location_id,
        temperature_c=weather.temperature_c if weather else None,
        humidity_pct=weather.humidity_pct if weather else None,
        wind_speed_kmh=weather.wind_speed_kmh if weather else None,
        pressure_hpa=weather.pressure_hpa if weather else None,
        cloud_cover_pct=weather.cloud_cover_pct if weather else None,
        solar_elevation_deg=astro.solar_elevation_deg if astro else None,
    )
    if not state:
        return None

    return AtmosphereSchema(
        european_aqi=state.european_aqi,
        european_aqi_label=state.european_aqi_label,
        us_aqi=state.us_aqi,
        pm2_5=state.pm2_5,
        pm10=state.pm10,
        ozone=state.ozone,
        nitrogen_dioxide=state.nitrogen_dioxide,
        sulphur_dioxide=state.sulphur_dioxide,
        carbon_monoxide=state.carbon_monoxide,
        dew_point_c=state.dew_point_c,
        feels_like_c=state.feels_like_c,
        uv_index=state.uv_index,
        air_density_kgm3=state.air_density_kgm3,
    )


# --- Climate ---

@router.get("/climate/{location_id}")
async def get_climate(location_id: int):
    """Get the climate profile for a location.

    Returns monthly temperature/precipitation normals, annual summaries,
    and Koppen climate classification. Data from Open-Meteo Archive API,
    cached permanently after first fetch.
    """
    world = get_world()
    loc = world.geography.get_location(location_id) if world.geography else None
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")

    if not hasattr(world, "climate") or world.climate is None:
        from world.climate import Climate
        world.climate = Climate()

    state = await world.climate.get_climate(location_id, loc.lat, loc.lng)
    result = state.to_dict()
    result["location_id"] = location_id
    result["location_name"] = loc.name
    return result


# --- Orbital ---

@router.get("/orbital", response_model=OrbitalSchema)
async def get_orbital():
    """Get Earth's current orbital state (distance, speed, position, season)."""
    world = get_world()
    state = world.orbital.compute(world.time.current_time)
    return OrbitalSchema(
        earth_sun_distance_km=state.earth_sun_distance_km,
        earth_sun_distance_au=state.earth_sun_distance_au,
        orbital_position_deg=state.orbital_position_deg,
        true_anomaly_deg=state.true_anomaly_deg,
        mean_anomaly_deg=state.mean_anomaly_deg,
        orbital_speed_kms=state.orbital_speed_kms,
        axial_tilt_deg=state.axial_tilt_deg,
        solar_declination_deg=state.solar_declination_deg,
        season=state.season,
        season_progress=state.season_progress,
        days_to_perihelion=state.days_to_perihelion,
        days_to_next_event=state.days_to_next_event,
        next_event=state.next_event,
        eccentricity=state.eccentricity,
        semi_major_axis_km=state.semi_major_axis_km,
    )


# --- Data Feeds ---

@router.get("/data-feeds/solar", response_model=SolarActivitySchema | None)
async def get_solar_activity():
    """Get current solar activity data from NOAA Space Weather."""
    world = get_world()
    solar = world.data_feeds.solar_activity
    if not solar:
        return None
    return SolarActivitySchema(
        kp_index=solar.kp_index,
        kp_category=solar.kp_category,
        solar_wind_speed_kms=solar.solar_wind_speed_kms,
        solar_wind_density=solar.solar_wind_density,
        solar_wind_temperature_k=solar.solar_wind_temperature_k,
        bz_gsm_nt=solar.bz_gsm_nt,
        bt_nt=solar.bt_nt,
        xray_flux=solar.xray_flux,
        xray_class=solar.xray_class,
        last_updated=solar.last_updated,
    )


# --- Time ---

@router.get("/time")
async def get_time():
    """Get the current world time (global UTC)."""
    world = get_world()
    return world.time.to_dict()


@router.get("/time/{location_id}")
async def get_time_at_location(location_id: int):
    """Get the current local time at a specific location.

    Returns local time, IANA timezone, offset, abbreviation, and season
    based on the location's real-world coordinates.
    """
    world = get_world()
    loc = world.geography.get_location(location_id) if world.geography else None
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")

    time_info = world.time.time_at(loc.lat, loc.lng)
    time_info["season"] = world.time.season_at(loc.lat)
    time_info["location_id"] = location_id
    time_info["location_name"] = loc.name
    return time_info


@router.get("/rotation")
async def get_rotation():
    """Get the current Earth rotation state.

    Returns GMST angle, sub-solar point (lat/lng where the sun is
    directly overhead), and solar declination. Everything a frontend
    needs to render globe rotation and day/night terminator.
    """
    from datetime import datetime, timezone
    from world.celestial import earth_rotation_state

    now = datetime.now(timezone.utc)
    return earth_rotation_state(now)


# --- Evaluation Snapshot ---

@router.get("/eval/snapshot", response_model=EvalSnapshotSchema)
async def eval_snapshot():
    """Lightweight endpoint for evaluation harnesses.

    Returns current tick, all agent summaries, earth proxy stats,
    and world config in a single call.
    """
    from datetime import datetime, timezone

    world = get_world()
    agents = await world.list_agents()
    time_dict = world.time.to_dict() if world.time else {}
    earth_proxy_stats = None
    if world.earth_proxy:
        earth_proxy_stats = world.earth_proxy.get_policy_stats()

    return EvalSnapshotSchema(
        tick=time_dict.get("tick_count", 0),
        wall_time=datetime.now(timezone.utc).isoformat(),
        is_running=world.is_running,
        location_count=world.geography.location_count if world.geography else 0,
        agent_count=len(agents),
        agents=agents,
        earth_proxy=earth_proxy_stats,
        config=world.config.model_dump() if world.config else None,
    )


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
