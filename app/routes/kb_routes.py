"""Knowledge Base routes."""

from starlette.routing import Route
from starlette.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.requests import Request
from sqlalchemy import select
from app.database import get_db, SessionLocal
from app.models import KnowledgeBase, WebCrawlSource
from app.templates import templates
from app.services.web_crawl_service import WebCrawlService
from app.services.ingest_service import IngestService
import uuid
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


def _list_kbs_sync():
    """Sync function to list knowledge bases."""
    db = next(get_db())
    try:
        result = db.execute(
            select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc())
        )
        return result.scalars().all()
    finally:
        db.close()


async def list_kbs(request: Request) -> HTMLResponse:
    """List all knowledge bases."""
    kbs = await asyncio.to_thread(_list_kbs_sync)

    return templates.TemplateResponse(
        "home.html", {"request": request, "knowledge_bases": kbs}
    )


def _create_kb_sync(name: str, description: str):
    """Sync function to create knowledge base."""
    db = next(get_db())
    existing = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.name == name,
        )
        .first()
    )
    if existing:
        raise ValueError(f"Knowledge base with name {name} already exists")
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
            return None, None, None

        # Get files for this KB
        from app.models import UploadedFile

        files_result = db.execute(
            select(UploadedFile)
            .where(UploadedFile.kb_id == kb.id)
            .order_by(UploadedFile.created_at.desc())
        )
        files = files_result.scalars().all()

        # Get web crawl sources for this KB
        urls_result = db.execute(
            select(WebCrawlSource)
            .where(WebCrawlSource.kb_id == kb.id)
            .order_by(WebCrawlSource.created_at.desc())
        )
        urls = urls_result.scalars().all()

        return kb, files, urls
    finally:
        db.close()


async def kb_detail(request: Request) -> HTMLResponse:
    """Knowledge base detail page."""
    kb_id = request.path_params.get("kb_id")
    if not kb_id:
        return RedirectResponse(url="/", status_code=303)

    kb, files, urls = await asyncio.to_thread(_kb_detail_sync, kb_id)

    if not kb:
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        "kb_detail.html", {"request": request, "kb": kb, "files": files, "urls": urls}
    )


async def add_urls(request: Request) -> JSONResponse:
    """Add URLs to a knowledge base."""
    kb_id = request.path_params.get("kb_id")
    if not kb_id:
        return JSONResponse({"error": "KB ID required"}, status_code=400)

    try:
        data = await request.json()
        urls = data.get("urls", [])
        if not urls:
            return JSONResponse(
                {"error": "At least one URL is required"}, status_code=400
            )

        descriptions = data.get("descriptions", [])
        crawl_immediately = data.get("crawl_immediately", False)

        db = SessionLocal()
        try:
            crawl_service = WebCrawlService(db)
            created_sources = await crawl_service.add_urls_to_kb(
                kb_id, urls, descriptions
            )

            result = {
                "message": f"Added {len(created_sources)} URLs",
                "sources": [
                    {
                        "id": str(source.id),
                        "url": source.url,
                        "title": source.title,
                        "description": source.description,
                        "is_indexed": source.is_indexed,
                    }
                    for source in created_sources
                ],
            }

            # If crawl_immediately is True, crawl and index the URLs
            if crawl_immediately and created_sources:
                ingest_service = IngestService(db)
                crawl_results = {}
                for source in created_sources:
                    crawl_result = await crawl_service.crawl_and_index_url(
                        str(source.id), ingest_service
                    )
                    crawl_results[str(source.id)] = crawl_result
                result["crawl_results"] = crawl_results

            return JSONResponse(result)
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error adding URLs: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def update_url(request: Request) -> JSONResponse:
    """Update a web crawl source."""
    kb_id = request.path_params.get("kb_id")
    url_id = request.path_params.get("url_id")

    if not kb_id or not url_id:
        return JSONResponse({"error": "KB ID and URL ID required"}, status_code=400)

    try:
        data = await request.json()
        url = data.get("url")
        description = data.get("description")

        db = SessionLocal()
        try:
            result = db.execute(
                select(WebCrawlSource).where(
                    WebCrawlSource.id == uuid.UUID(url_id),
                    WebCrawlSource.kb_id == uuid.UUID(kb_id),
                )
            )
            source = result.scalar_one_or_none()

            if not source:
                return JSONResponse({"error": "Source not found"}, status_code=404)

            if url:
                source.url = url.strip()
            if description is not None:
                source.description = description.strip() if description else None

            db.commit()
            db.refresh(source)

            return JSONResponse(
                {
                    "message": "URL updated successfully",
                    "source": {
                        "id": str(source.id),
                        "url": source.url,
                        "title": source.title,
                        "description": source.description,
                        "is_indexed": source.is_indexed,
                    },
                }
            )
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error updating URL: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def delete_url(request: Request) -> JSONResponse:
    """Delete a web crawl source."""
    kb_id = request.path_params.get("kb_id")
    url_id = request.path_params.get("url_id")

    if not kb_id or not url_id:
        return JSONResponse({"error": "KB ID and URL ID required"}, status_code=400)

    try:
        db = SessionLocal()
        try:
            result = db.execute(
                select(WebCrawlSource).where(
                    WebCrawlSource.id == uuid.UUID(url_id),
                    WebCrawlSource.kb_id == uuid.UUID(kb_id),
                )
            )
            source = result.scalar_one_or_none()

            if not source:
                return JSONResponse({"error": "Source not found"}, status_code=404)

            db.delete(source)
            db.commit()

            return JSONResponse({"message": "URL deleted successfully"})
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error deleting URL: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def recrawl_url(request: Request) -> JSONResponse:
    """Recrawl and re-index a URL."""
    kb_id = request.path_params.get("kb_id")
    url_id = request.path_params.get("url_id")

    if not kb_id or not url_id:
        return JSONResponse({"error": "KB ID and URL ID required"}, status_code=400)

    try:
        db = next(get_db())
        try:
            crawl_service = WebCrawlService(db)

            result = await crawl_service.crawl_and_index_url(source_id=url_id)

            if result.get("success"):
                return JSONResponse(
                    {
                        "message": result.get(
                            "message", "URL crawled and indexed successfully"
                        ),
                        "chunks": result.get("chunks", 0),
                    }
                )
            else:
                return JSONResponse(
                    {"error": result.get("error", "Failed to crawl URL")},
                    status_code=500,
                )
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error recrawling URL: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def crawl_all_urls(request: Request) -> JSONResponse:
    """Crawl and index all URLs for a knowledge base."""
    kb_id = request.path_params.get("kb_id")

    if not kb_id:
        return JSONResponse({"error": "KB ID required"}, status_code=400)

    try:
        db = SessionLocal()
        try:
            # Get all URLs for this KB
            result = db.execute(
                select(WebCrawlSource).where(WebCrawlSource.kb_id == uuid.UUID(kb_id))
            )
            sources = result.scalars().all()

            if not sources:
                return JSONResponse({"message": "No URLs to crawl", "results": {}})

            crawl_service = WebCrawlService(db)

            source_ids = [str(source.id) for source in sources]
            results = await crawl_service.crawl_and_index_multiple_urls(
                source_ids=source_ids,
            )

            # Format results
            formatted_results = {}
            success_count = 0
            for source_id, result in results.items():
                formatted_results[source_id] = {
                    "success": result.get("success", False),
                    "message": result.get("message")
                    or result.get("error", "Unknown error"),
                    "chunks": result.get("chunks", 0),
                }
                if result.get("success"):
                    success_count += 1

            return JSONResponse(
                {
                    "message": f"Crawled {success_count}/{len(sources)} URLs successfully",
                    "results": formatted_results,
                }
            )
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error crawling all URLs: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


kb_routes = [
    Route("/", list_kbs, methods=["GET"]),
    Route("/kb/create", create_kb, methods=["POST"]),
    Route("/kb/{kb_id}/delete", delete_kb, methods=["POST"]),
    Route("/kb/{kb_id}", kb_detail, methods=["GET"]),
    Route("/kb/{kb_id}/crawl", add_urls, methods=["POST"]),
    Route("/kb/{kb_id}/urls/{url_id}", update_url, methods=["PUT"]),
    Route("/kb/{kb_id}/urls/{url_id}", delete_url, methods=["DELETE"]),
    Route("/kb/{kb_id}/urls/{url_id}/recrawl", recrawl_url, methods=["POST"]),
    Route("/kb/{kb_id}/urls/crawl-all", crawl_all_urls, methods=["POST"]),
]
