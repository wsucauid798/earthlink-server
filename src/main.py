import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router, set_world
from api.ws import router as ws_router, on_tick
from db.engine import dispose_engine, engine
from db.models import Base
from world import World

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist, load and start the world
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    world = World()
    await world.load()
    world.on_tick(on_tick)
    set_world(world)
    if not world.is_running:
        await world.start()

    logger.info("EarthLink server ready")

    yield

    # Shutdown
    if world.is_running:
        await world.pause()
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
app.include_router(ws_router)
