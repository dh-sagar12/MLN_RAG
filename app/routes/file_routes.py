"""File upload routes."""
import logging
from starlette.routing import Route
from starlette.responses import RedirectResponse, JSONResponse
from starlette.requests import Request
from sqlalchemy import select
from app.database import get_db
from app.models import KnowledgeBase, UploadedFile
from app.services.file_service import FileService
from app.services.ingest_service import IngestService
import uuid
import asyncio

logger = logging.getLogger(__name__)


def _verify_kb_sync(kb_id: str):
    """Sync function to verify KB exists."""
    db = next(get_db())
    try:
        result = db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == uuid.UUID(kb_id))
        )
        return result.scalar_one_or_none()
    finally:
        db.close()


def _save_file_record_sync(kb_id: uuid.UUID, file_name: str, file_type: str, file_size: int, file_path: str):
    """Sync function to save file record."""
    db = next(get_db())
    try:
        uploaded_file = UploadedFile(
            kb_id=kb_id,
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            file_path=file_path
        )
        db.add(uploaded_file)
        db.commit()
        db.refresh(uploaded_file)
        return uploaded_file
    finally:
        db.close()


async def upload_file(request: Request) -> RedirectResponse:
    """Upload a file to a knowledge base."""
    kb_id = request.path_params.get("kb_id")
    logger.info(f"File upload requested for KB: {kb_id}")
    
    if not kb_id:
        logger.warning("No KB ID provided in upload request")
        return RedirectResponse(url="/", status_code=303)
    
    # Verify KB exists
    kb = await asyncio.to_thread(_verify_kb_sync, kb_id)
    if not kb:
        logger.warning(f"KB not found: {kb_id}")
        return RedirectResponse(url="/", status_code=303)
    
    form = await request.form()
    file = form.get("file")
    
    if not file or not hasattr(file, "filename"):
        logger.warning(f"No file provided in upload request for KB: {kb_id}")
        return RedirectResponse(url=f"/kb/{kb_id}", status_code=303)
    
    logger.info(f"Processing file upload: {file.filename} for KB: {kb_id}")
    
    # Save file (async file I/O)
    file_service = FileService()
    file_path, file_size = await file_service.save_file(file, kb_id)
    file_type = file_service.get_file_type(file.filename, file.content_type)
    logger.info(f"File saved: {file_path} (size: {file_size} bytes, type: {file_type})")
    
    # Save database record
    await asyncio.to_thread(_save_file_record_sync, kb.id, file.filename, file_type, file_size, file_path)
    logger.info(f"File record saved to database: {file.filename}")
    
    # Process file in background
    asyncio.create_task(process_file_background(kb.id, file_path, file_type))
    logger.info(f"Background processing task started for file: {file_path}")
    
    return RedirectResponse(url=f"/kb/{kb_id}", status_code=303)


async def process_file_background(kb_id: uuid.UUID, file_path: str, file_type: str) -> None:
    """Background task to process file."""
    logger.info(f"Starting background processing for file: {file_path} (KB: {kb_id})")
    db = next(get_db())
    try:
        ingest_service = IngestService(db)
        await ingest_service.process_file(kb_id, file_path, file_type)
        logger.info(f"Successfully processed file: {file_path}")
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
        raise
    finally:
        db.close()


def _process_kb_verify_sync(kb_id: str):
    """Sync function to verify KB for processing."""
    db = next(get_db())
    try:
        result = db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == uuid.UUID(kb_id))
        )
        return result.scalar_one_or_none()
    finally:
        db.close()


async def process_kb(request: Request) -> JSONResponse:
    """Process all files in a knowledge base (create embeddings)."""
    kb_id = request.path_params.get("kb_id")
    logger.info(f"Process KB request received for KB: {kb_id}")
    
    if not kb_id:
        logger.warning("Process KB request missing KB ID")
        return JSONResponse({"error": "KB ID required"}, status_code=400)
    
    # Verify KB exists
    kb = await asyncio.to_thread(_process_kb_verify_sync, kb_id)
    if not kb:
        logger.warning(f"KB not found for processing: {kb_id}")
        return JSONResponse({"error": "KB not found"}, status_code=404)
    
    # Process in background
    asyncio.create_task(process_kb_background(kb.id))
    logger.info(f"Background processing task started for KB: {kb_id}")
    
    return JSONResponse({"message": "Processing started"})


async def process_kb_background(kb_id: uuid.UUID) -> None:
    """Background task to process all files in KB."""
    logger.info(f"Starting background processing for KB: {kb_id}")
    db = next(get_db())
    try:
        ingest_service = IngestService(db)
        await ingest_service.create_embeddings_for_kb(kb_id)
        logger.info(f"Successfully processed all files for KB: {kb_id}")
    except Exception as e:
        logger.error(f"Error processing KB {kb_id}: {str(e)}", exc_info=True)
        raise
    finally:
        db.close()


file_routes = [
    Route("/kb/{kb_id}/upload", upload_file, methods=["POST"]),
    Route("/kb/{kb_id}/process", process_kb, methods=["POST"]),
]
