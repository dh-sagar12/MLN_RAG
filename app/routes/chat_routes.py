"""Chat/query routes."""

import logging
from starlette.routing import Route
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.requests import Request
from sqlalchemy import select, desc
from app.database import get_db
from app.services.rag_service import RAGService
from app.models import ChatSession, ChatMessage
from app.templates import templates
import asyncio
import uuid

logger = logging.getLogger(__name__)


def _get_chat_sessions_sync():
    """Sync function to get all chat sessions."""
    db = next(get_db())
    try:
        result = db.execute(select(ChatSession).order_by(desc(ChatSession.updated_at)))
        return result.scalars().all()
    finally:
        db.close()


def _create_chat_session_sync():
    """Sync function to create a new chat session."""
    db = next(get_db())
    try:
        session = ChatSession()
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    finally:
        db.close()


def _get_chat_session_sync(session_id: str):
    """Sync function to get a chat session with messages."""
    db = next(get_db())
    try:
        result = db.execute(
            select(ChatSession).where(ChatSession.id == uuid.UUID(session_id))
        )
        session = result.scalar_one_or_none()
        if session:
            # Load messages
            _ = session.messages
        return session
    finally:
        db.close()


def _save_message_sync(session_id: str, role: str, content: str):
    """Sync function to save a chat message."""
    db = next(get_db())
    try:
        message = ChatMessage(
            session_id=uuid.UUID(session_id), role=role, content=content
        )
        db.add(message)

        # Update session title if it's the first user message
        session = db.get(ChatSession, uuid.UUID(session_id))
        if not session.title and role == "user":
            session.title = content[:50] + ("..." if len(content) > 50 else "")

        db.commit()
        db.refresh(message)
        return message
    finally:
        db.close()


def _get_chat_history_sync(session_id: str, k: int = 10):
    """Sync function to get last k messages from chat history."""
    db = next(get_db())
    try:
        result = db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == uuid.UUID(session_id))
            .order_by(desc(ChatMessage.created_at))
            .limit(k)
        )
        messages = result.scalars().all()
        # Reverse to get chronological order
        messages.reverse()
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    finally:
        db.close()


def _delete_chat_session_sync(session_id: str):
    """Sync function to delete a chat session."""
    db = next(get_db())
    try:
        session = db.get(ChatSession, uuid.UUID(session_id))
        if session:
            db.delete(session)
            db.commit()
            return True
        return False
    finally:
        db.close()


def _query_with_history_sync(
    query_text: str, top_k: int, session_id: str = None, history_k: int = 10
):
    """Sync function to query RAG system with chat history."""
    db = next(get_db())
    try:
        rag_service = RAGService(db)

        # Get chat history if session_id provided
        chat_history = None
        if session_id:
            chat_history = _get_chat_history_sync(session_id, history_k)
        result = rag_service.query(
            query_text=query_text,
            top_k=top_k,
            chat_history=chat_history,
        )
        return result
    finally:
        db.close()


async def chat_page(request: Request) -> HTMLResponse:
    """Chat page."""
    # Get all chat sessions
    sessions = await asyncio.to_thread(_get_chat_sessions_sync)

    # Get current session from query params
    session_id = request.query_params.get("session_id")
    current_session = None
    if session_id:
        current_session = await asyncio.to_thread(_get_chat_session_sync, session_id)

    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "sessions": sessions, "current_session": current_session},
    )


async def create_session(request: Request) -> JSONResponse:
    """Create a new chat session."""
    try:
        session = await asyncio.to_thread(_create_chat_session_sync)
        logger.info(f"Created new chat session: {session.id}")
        return JSONResponse(
            {
                "session_id": str(session.id),
                "title": session.title,
                "created_at": session.created_at.isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_session(request: Request) -> JSONResponse:
    """Get a chat session with messages."""
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse({"error": "Session ID required"}, status_code=400)

    try:
        session = await asyncio.to_thread(_get_chat_session_sync, session_id)
        if not session:
            return JSONResponse({"error": "Session not found"}, status_code=404)

        messages = [
            {
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
            }
            for msg in session.messages
        ]

        return JSONResponse(
            {
                "session_id": str(session.id),
                "title": session.title,
                "created_at": session.created_at.isoformat(),
                "messages": messages,
            }
        )
    except Exception as e:
        logger.error(f"Error getting chat session: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def delete_session(request: Request) -> JSONResponse:
    """Delete a chat session."""
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse({"error": "Session ID required"}, status_code=400)

    try:
        deleted = await asyncio.to_thread(_delete_chat_session_sync, session_id)
        if deleted:
            logger.info(f"Deleted chat session: {session_id}")
            return JSONResponse({"message": "Session deleted"})
        else:
            return JSONResponse({"error": "Session not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error deleting chat session: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def list_sessions(request: Request) -> JSONResponse:
    """List all chat sessions."""
    try:
        sessions = await asyncio.to_thread(_get_chat_sessions_sync)
        sessions_data = [
            {
                "id": str(session.id),
                "title": session.title or "New Chat",
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
            }
            for session in sessions
        ]
        return JSONResponse({"sessions": sessions_data})
    except Exception as e:
        logger.error(f"Error listing chat sessions: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def query_api(request: Request) -> JSONResponse:
    """Query API endpoint with chat history support."""
    try:
        body = await request.json()
        query_text = body.get("query", "").strip()
        top_k = body.get("top_k", 5)
        session_id = body.get("session_id")
        history_k = body.get("history_k", 10)  # Number of previous messages to include

        logger.info(
            f"Query API request: query='{query_text[:50]}...', session_id={session_id}, top_k={top_k}, history_k={history_k}"
        )

        if not query_text:
            logger.warning("Query API request missing query text")
            return JSONResponse({"error": "Query is required"}, status_code=400)

        # Execute query with history
        logger.debug("Executing query in thread pool...")
        result = await asyncio.to_thread(
            _query_with_history_sync, query_text, top_k, session_id, history_k
        )

        # Save messages to database if session_id provided
        if session_id:
            await asyncio.to_thread(_save_message_sync, session_id, "user", query_text)
            await asyncio.to_thread(
                _save_message_sync, session_id, "assistant", result["answer"]
            )
            logger.info(f"Saved messages to session: {session_id}")

        logger.info(f"Query completed successfully")
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


chat_routes = [
    Route("/chat", chat_page, methods=["GET"]),
    Route("/api/chat/sessions", list_sessions, methods=["GET"]),
    Route("/api/chat/sessions", create_session, methods=["POST"]),
    Route("/api/chat/sessions/{session_id}", get_session, methods=["GET"]),
    Route("/api/chat/sessions/{session_id}", delete_session, methods=["DELETE"]),
    Route("/api/query", query_api, methods=["POST"]),
]
