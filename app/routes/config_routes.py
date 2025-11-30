"""Configuration management routes."""

import logging
from starlette.routing import Route
from starlette.responses import HTMLResponse, JSONResponse
from starlette.requests import Request
from sqlalchemy import select
from app.database import get_db, SessionLocal
from app.models import Configuration
from app.services.config_service import ConfigService
from app.templates import templates
import asyncio

logger = logging.getLogger(__name__)


def _get_all_configs_sync(category: str = None):
    """Sync function to get all configurations."""
    db = next(get_db())
    try:
        return ConfigService.get_all_configs(db, category)
    finally:
        db.close()


def _get_config_sync(key: str):
    """Sync function to get a single configuration."""
    db = next(get_db())
    try:
        config = db.execute(
            select(Configuration).where(Configuration.key == key)
        ).scalar_one_or_none()
        
        if config:
            return {
                "key": config.key,
                "value": config.get_typed_value(),
                "type": config.value_type,
                "description": config.description,
                "category": config.category,
            }
        return None
    finally:
        db.close()


def _set_config_sync(key: str, value, description: str = None):
    """Sync function to set a configuration value."""
    db = SessionLocal()
    try:
        config = ConfigService.set_config(db, key, value, description)
        return {
            "key": config.key,
            "value": config.get_typed_value(),
            "type": config.value_type,
            "description": config.description,
            "category": config.category,
        }
    finally:
        db.close()


def _set_multiple_configs_sync(configs: dict):
    """Sync function to set multiple configuration values."""
    db = SessionLocal()
    try:
        results = {}
        for key, value in configs.items():
            config = ConfigService.set_config(db, key, value)
            results[key] = {
                "key": config.key,
                "value": config.get_typed_value(),
                "type": config.value_type,
            }
        return results
    finally:
        db.close()


async def config_page(request: Request) -> HTMLResponse:
    """Configuration management page."""
    category = request.query_params.get("category", "")
    configs = await asyncio.to_thread(_get_all_configs_sync, category if category else None)
    
    # Organize configs by category
    organized_configs = {}
    for key, config_data in configs.items():
        cat = config_data["category"]
        if cat not in organized_configs:
            organized_configs[cat] = []
        organized_configs[cat].append({
            "key": key,
            "value": config_data["value"],
            "type": config_data["type"],
            "description": config_data.get("description", ""),
            "category": cat,
        })
    
    return templates.TemplateResponse(
        "config.html",
        {
            "request": request,
            "configs": organized_configs,
            "current_category": category,
        }
    )


async def get_configs_api(request: Request) -> JSONResponse:
    """Get all configurations via API."""
    category = request.query_params.get("category")
    try:
        configs = await asyncio.to_thread(_get_all_configs_sync, category)
        return JSONResponse({"configs": configs})
    except Exception as e:
        logger.error(f"Error getting configurations: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_config_api(request: Request) -> JSONResponse:
    """Get a single configuration via API."""
    key = request.path_params.get("key")
    if not key:
        return JSONResponse({"error": "Configuration key required"}, status_code=400)
    
    try:
        config = await asyncio.to_thread(_get_config_sync, key)
        if config:
            return JSONResponse(config)
        else:
            return JSONResponse({"error": "Configuration not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def set_config_api(request: Request) -> JSONResponse:
    """Set a configuration value via API."""
    try:
        body = await request.json()
        key = body.get("key")
        value = body.get("value")
        description = body.get("description")
        
        if not key:
            return JSONResponse({"error": "Configuration key required"}, status_code=400)
        
        if value is None:
            return JSONResponse({"error": "Configuration value required"}, status_code=400)
        
        config = await asyncio.to_thread(_set_config_sync, key, value, description)
        logger.info(f"Updated configuration: {key} = {value}")
        return JSONResponse(config)
    except Exception as e:
        logger.error(f"Error setting configuration: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def set_multiple_configs_api(request: Request) -> JSONResponse:
    """Set multiple configuration values via API."""
    try:
        body = await request.json()
        configs = body.get("configs", {})
        
        if not configs:
            return JSONResponse({"error": "Configurations object required"}, status_code=400)
        
        results = await asyncio.to_thread(_set_multiple_configs_sync, configs)
        logger.info(f"Updated {len(results)} configurations")
        return JSONResponse({"configs": results, "count": len(results)})
    except Exception as e:
        logger.error(f"Error setting configurations: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def reset_config_api(request: Request) -> JSONResponse:
    """Reset a configuration to its default value."""
    key = request.path_params.get("key")
    if not key:
        return JSONResponse({"error": "Configuration key required"}, status_code=400)
    
    try:
        db = SessionLocal()
        try:
            default_data = ConfigService.DEFAULT_CONFIG.get(key)
            if not default_data:
                return JSONResponse({"error": "No default value found for this configuration"}, status_code=404)
            
            config = ConfigService.set_config(db, key, default_data["value"], default_data.get("description"))
            logger.info(f"Reset configuration to default: {key}")
            return JSONResponse({
                "key": config.key,
                "value": config.get_typed_value(),
                "type": config.value_type,
                "description": config.description,
            })
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error resetting configuration: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


config_routes = [
    Route("/config", config_page, methods=["GET"]),
    Route("/api/config", get_configs_api, methods=["GET"]),
    Route("/api/config/{key}", get_config_api, methods=["GET"]),
    Route("/api/config", set_config_api, methods=["POST"]),
    Route("/api/config/bulk", set_multiple_configs_api, methods=["POST"]),
    Route("/api/config/{key}/reset", reset_config_api, methods=["POST"]),
]

