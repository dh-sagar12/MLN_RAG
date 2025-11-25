"""Knowledge Base routes."""
from starlette.routing import Route
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request
from sqlalchemy import select
from app.database import get_db
from app.models import KnowledgeBase
from app.templates import templates
import uuid
import asyncio


def _list_kbs_sync():
    """Sync function to list knowledge bases."""
    db = next(get_db())
    try:
        result = db.execute(select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc()))
        return result.scalars().all()
    finally:
        db.close()


async def list_kbs(request: Request) -> HTMLResponse:
    """List all knowledge bases."""
    kbs = await asyncio.to_thread(_list_kbs_sync)
    
    return templates.TemplateResponse("home.html", {
        "request": request,
        "knowledge_bases": kbs
    })


def _create_kb_sync(name: str, description: str):
    """Sync function to create knowledge base."""
    db = next(get_db())
    try:
        if name:
            kb = KnowledgeBase(name=name, description=description)
            db.add(kb)
            db.commit()
            db.refresh(kb)
    finally:
        db.close()


async def create_kb(request: Request) -> RedirectResponse:
    """Create a new knowledge base."""
    form = await request.form()
    name = form.get("name", "").strip()
    description = form.get("description", "").strip()
    
    await asyncio.to_thread(_create_kb_sync, name, description)
    
    return RedirectResponse(url="/", status_code=303)


def _delete_kb_sync(kb_id: str):
    """Sync function to delete knowledge base."""
    db = next(get_db())
    try:
        if kb_id:
            result = db.execute(
                select(KnowledgeBase).where(KnowledgeBase.id == uuid.UUID(kb_id))
            )
            kb = result.scalar_one_or_none()
            if kb:
                db.delete(kb)
                db.commit()
    finally:
        db.close()


async def delete_kb(request: Request) -> RedirectResponse:
    """Delete a knowledge base."""
    kb_id = request.path_params.get("kb_id")
    await asyncio.to_thread(_delete_kb_sync, kb_id)
    
    return RedirectResponse(url="/", status_code=303)


def _kb_detail_sync(kb_id: str):
    """Sync function to get knowledge base detail."""
    db = next(get_db())
    try:
        result = db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == uuid.UUID(kb_id))
        )
        kb = result.scalar_one_or_none()
        
        if not kb:
            return None, None
        
        # Get files for this KB
        from app.models import UploadedFile
        files_result = db.execute(
            select(UploadedFile).where(UploadedFile.kb_id == kb.id).order_by(UploadedFile.created_at.desc())
        )
        files = files_result.scalars().all()
        
        return kb, files
    finally:
        db.close()


async def kb_detail(request: Request) -> HTMLResponse:
    """Knowledge base detail page."""
    kb_id = request.path_params.get("kb_id")
    if not kb_id:
        return RedirectResponse(url="/", status_code=303)
    
    kb, files = await asyncio.to_thread(_kb_detail_sync, kb_id)
    
    if not kb:
        return RedirectResponse(url="/", status_code=303)
    
    return templates.TemplateResponse("kb_detail.html", {
        "request": request,
        "kb": kb,
        "files": files
    })


kb_routes = [
    Route("/", list_kbs, methods=["GET"]),
    Route("/kb/create", create_kb, methods=["POST"]),
    Route("/kb/{kb_id}/delete", delete_kb, methods=["POST"]),
    Route("/kb/{kb_id}", kb_detail, methods=["GET"]),
]
