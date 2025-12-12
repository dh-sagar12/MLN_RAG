"""File upload routes."""

import logging
from typing import List
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


def _save_file_record_sync(
    kb_id: uuid.UUID, file_name: str, file_type: str, file_size: int, file_path: str
):
    """Sync function to save file record."""
    db = next(get_db())
    try:
        uploaded_file = UploadedFile(
            kb_id=kb_id,
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            file_path=file_path,
        )
        db.add(uploaded_file)
        db.commit()
        db.refresh(uploaded_file)
        return uploaded_file
    finally:
        db.close()


async def process_files_background(kb_id: uuid.UUID, file_paths: List[UploadedFile]) -> None:
    """Background task to process multiple files."""
    logger.info(
        f"Starting background processing for {len(file_paths)} files (KB: {kb_id})"
    )
    db = next(get_db())
    try:
        ingest_service = IngestService(db)
        await ingest_service.process_multiple_files(
            kb_id=kb_id,
            file_paths=file_paths,
        )
        logger.info(f"Successfully processed files")
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}", exc_info=True)
        # Status updates are handled inside process_files
    finally:
        db.close()


async def upload_file(request: Request) -> RedirectResponse:
    """Upload multiple files to a knowledge base."""
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
    # Check for 'files' (multiple) or 'file' (single)
    files = form.getlist("files")
    if not files:
        single_file = form.get("file")
        if single_file:
            files = [single_file]

    valid_files = [f for f in files if hasattr(f, "filename") and f.filename]

    if not valid_files:
        logger.warning(f"No files provided in upload request for KB: {kb_id}")
        return RedirectResponse(url=f"/kb/{kb_id}", status_code=303)

    logger.info(f"Processing {len(valid_files)} files for KB: {kb_id}")

    file_service = FileService()
    saved_file_paths = []

    for file in valid_files:
        try:
            # Save file (async file I/O)
            file_path, file_size = await file_service.save_file(file, kb_id)
            file_type = file_service.get_file_type(file.filename, file.content_type)
            logger.info(
                f"File saved: {file_path} (size: {file_size} bytes, type: {file_type})"
            )

            # Save database record
            uploaded_file  = await asyncio.to_thread(
                _save_file_record_sync,
                kb.id,
                file.filename,
                file_type,
                file_size,
                file_path,
            )
            logger.info(f"File record saved to database: {file.filename}")
            saved_file_paths.append(uploaded_file)
        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {e}")

    # Process files in background
    if saved_file_paths:
        asyncio.create_task(process_files_background(kb.id, saved_file_paths))
        logger.info(
            f"Background processing task started for {len(saved_file_paths)} files"
        )

    return RedirectResponse(url=f"/kb/{kb_id}", status_code=303)


# NOTE: CURRENTLY NOT IN USED AS WE ARE USING THE `process_files_background` FUNCTION INSTEAD
async def process_file_background(
    kb_id: uuid.UUID, file_path: str, file_type: str
) -> None:
    """Background task to process file."""
    logger.info(f"Starting background processing for file: {file_path} (KB: {kb_id})")
    db = next(get_db())
    try:
        ingest_service = IngestService(db)
        await ingest_service.process_file(
            kb_id=kb_id,
            file_path=file_path,
            file_type=file_type,
        )
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


async def upload_emails(request: Request) -> JSONResponse:
    """Upload emails to a knowledge base."""
    kb_id = request.path_params.get("kb_id")
    logger.info(f"Email upload requested for KB: {kb_id}")

    if not kb_id:
        logger.warning("No KB ID provided in upload request")
        return JSONResponse({"error": "KB ID required"}, status_code=400)
    
    db = next(get_db())
    try:
        ingest_service = IngestService(db)
        asyncio.create_task(ingest_service.process_email(kb_id=kb_id))
        logger.info(f"Successfully processed email for KB: {kb_id}")
    except Exception as e:
        logger.error(f"Error processing email for KB: {kb_id}: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        db.close()

    return JSONResponse({"message": "Email processing started"})
    


file_routes = [
    Route("/kb/{kb_id}/upload", upload_file, methods=["POST"]),
    Route("/kb/{kb_id}/process", process_kb, methods=["POST"]),
    Route("/kb/{kb_id}/upload-emails", upload_emails, methods=["GET"]),
]
