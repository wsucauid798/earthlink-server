import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router, set_world
from api.wt import on_tick_wt, start_wt_fanout, start_wt_server, stop_wt_fanout, stop_wt_server
from db.engine import dispose_engine, engine, async_session
from db.models import Base
from world import World
from world.config import WorldConfig
from config import settings
from sqlalchemy import text

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist, load and start the world
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    # Alembic FIRST — must run before create_all because some columns
    # (e.g. agent_facts.embedding = vector(1024)) depend on the pgvector
    # extension being installed, which is done by migration c5e7a09f8b21.
    # Retries handle the brief window when Postgres is up but not ready.
    for attempt in range(1, 6):
        try:
            from alembic.config import Config as _AlembicConfig
            from alembic import command as _alembic_command
            cfg = _AlembicConfig("/app/alembic.ini")
            cfg.set_main_option("script_location", "/app/src/db/migrations")
            # Override the hard-coded URL in alembic.ini with the actual
            # runtime DB URL from settings (in-container DNS, prod creds).
            cfg.set_main_option("sqlalchemy.url", settings.database_url)
            await asyncio.to_thread(_alembic_command.upgrade, cfg, "head")
            logger.info("alembic migrations: at head")
            break
        except Exception as exc:
            if attempt == 5:
                raise
            logger.warning("alembic upgrade failed (%s), retrying in %ds...", exc, attempt * 2)
            await asyncio.sleep(attempt * 2)

    # create_all is the safety net for any model that isn't covered by an
    # alembic migration (e.g. dev-only tables, or one defined before its
    # migration is written). Idempotent — only creates missing tables.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Auto-seed geography. `seed_geography()` (run) is internally idempotent:
    # it queries which countries are already present (via metadata->>'country_code'
    # in fetch_geography.py:444) and only fetches the missing ones. Cheap on a
    # populated DB, full seed on an empty one, incremental seed when new
    # countries are added to COUNTRIES_TO_FETCH.
    from data_acquisition.fetch_geography import run as seed_geography
    logger.info("Checking geography seed (run() will skip countries already present)")
    await seed_geography()

    world = World(WorldConfig(agent_count=settings.agent_count))
    await world.load()
    world.on_tick(on_tick_wt)
    set_world(world)
    if not world.is_running:
        await world.start()
    # S90: fan out tick broadcasts across WT-serving processes via Redis Streams.
    # Reuses the EarthProxy Redis connection; no-op (in-process) when Redis is absent.
    await start_wt_fanout(world.earth_proxy.redis if world.earth_proxy else None)
    await start_wt_server()

    logger.info("EarthLink server ready")

    yield

    # Shutdown
    if world.is_running:
        await world.pause()
    await stop_wt_fanout()
    await stop_wt_server()
    if world.earth_proxy:
        await world.earth_proxy.disconnect_redis()
    # Shut down Ray if it was initialised
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shut down")
    except ImportError:
        pass
    await dispose_engine()


app = FastAPI(
    title="EarthLink Server",
    description="A symbolic virtual world — built from real Earth data",
    version="0.0.2",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
