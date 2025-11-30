"""Main application entry point."""
import logging
import sys
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from starlette.routing import Mount
from pathlib import Path
from app.routes import kb_routes, file_routes, chat_routes
from app.routes import performance_routes, config_routes
from app.database import init_db
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


async def startup():
    """Initialize database on startup."""
    logger.info("Starting application...")
    # Run sync init_db in thread pool
    try:
        await asyncio.to_thread(init_db)
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}", exc_info=True)
        raise


def create_app() -> Starlette:
    """Create and configure Starlette application."""
    app = Starlette(
        routes=[
            *kb_routes.kb_routes,
            *file_routes.file_routes,
            *chat_routes.chat_routes,
            *performance_routes.performance_routes,
            *config_routes.config_routes,
            Mount("/static", app=StaticFiles(directory=str(Path(__file__).parent / "static")), name="static"),
        ],
        middleware=[
            Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        ],
        on_startup=[startup]
    )
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    from app.config import settings
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)

