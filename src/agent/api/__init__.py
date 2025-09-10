from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..shared.logging.logger import Logger
from ..shared.settings.settings import settings
from .dependencies.service import cleanup_services, initialize_services
from .routers import agent

logger = Logger.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Stock Assistant API...")
    logger.info("=" * 70)

    # Initialize services
    logger.info(
        "Initializing global services (Redis session manager and ServiceManager)..."
    )
    await initialize_services()
    logger.info("Global services initialized")
    logger.info("=" * 70)

    # Warm up singleton services
    from ..shared.singletons.service_manager import ServiceManager

    logger.info("Warming up ServiceManager...")
    service_manager = await ServiceManager.get_instance()
    await service_manager.get_embeddings()  # Warm up
    await service_manager.get_vector_store()  # Warm up
    logger.info("ServiceManager warmed up")
    logger.info("=" * 70)
    logger.info("Core services warmed up")

    yield

    # Shutdown
    logger.info("Shutting down Stock Assistant API...")
    logger.info("=" * 70)
    logger.info("Cleaning up global services...")
    await cleanup_services()
    logger.info("Global services cleaned up")
    logger.info("=" * 70)


def create_app() -> FastAPI:
    """Create FastAPI application"""

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="Vietnamese Stock Market ReAct Agent API",
        debug=settings.app.debug,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(agent.router)

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "Welcome to Milano-Stock Agent",
            "version": settings.app.version,
            "docs": "/docs",
        }

    return app
