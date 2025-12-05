"""Chat/query routes."""

import logging
from starlette.routing import Route
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.requests import Request
from sqlalchemy import select, desc
from app.database import get_db
from app.services.config_service import ConfigService
from app.services.rag_service import RAGService
from app.models import ChatSession, ChatMessage, DraftResponse
from app.templates import templates
import asyncio
import uuid
import json

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


def _get_chat_history_sync(
    session_id: str,
    k: int = 20,
):
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
    query_text: str,
    top_k: int,
    session_id: str = None,
    history_k: int = None,
    channel: str = "email",
    similarity_threshold: float = None,
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
            channel=channel,
            similarity_threshold=similarity_threshold,
        )
        return result
    finally:
        db.close()


# ==================== Co-pilot Mode Functions ====================

def _get_session_copilot_mode_sync(session_id: str) -> bool:
    """Get co-pilot mode status for a session."""
    db = next(get_db())
    try:
        session = db.get(ChatSession, uuid.UUID(session_id))
        return session.copilot_enabled if session else False
    finally:
        db.close()


def _toggle_copilot_mode_sync(session_id: str, enabled: bool):
    """Toggle co-pilot mode for a session."""
    db = next(get_db())
    try:
        session = db.get(ChatSession, uuid.UUID(session_id))
        if session:
            session.copilot_enabled = enabled
            db.commit()
            return True
        return False
    finally:
        db.close()


def _create_draft_sync(session_id: str, original_query: str, draft_content: str, sources_data: dict = None):
    """Create a new draft response."""
    db = next(get_db())
    try:
        # First, discard any existing active drafts for this session
        existing_drafts = db.execute(
            select(DraftResponse).where(
                DraftResponse.session_id == uuid.UUID(session_id),
                DraftResponse.status == "active"
            )
        ).scalars().all()
        
        for draft in existing_drafts:
            draft.status = "discarded"
        
        # Create new draft
        draft = DraftResponse(
            session_id=uuid.UUID(session_id),
            original_query=original_query,
            current_draft=draft_content,
            refinement_history=[],
            sources_data=sources_data or {},
            status="active"
        )
        db.add(draft)
        db.commit()
        db.refresh(draft)
        return draft
    finally:
        db.close()


def _get_active_draft_sync(session_id: str):
    """Get the active draft for a session."""
    db = next(get_db())
    try:
        result = db.execute(
            select(DraftResponse).where(
                DraftResponse.session_id == uuid.UUID(session_id),
                DraftResponse.status == "active"
            ).order_by(desc(DraftResponse.created_at))
        )
        return result.scalar_one_or_none()
    finally:
        db.close()


def _refine_draft_sync(draft_id: str, refinement_request: str, new_content: str):
    """Update a draft with refinement."""
    db = next(get_db())
    try:
        draft = db.get(DraftResponse, uuid.UUID(draft_id))
        if draft and draft.status == "active":
            # Add to refinement history
            history = list(draft.refinement_history) if draft.refinement_history else []
            history.append({
                "role": "user",
                "content": refinement_request,
            })
            history.append({
                "role": "assistant",
                "content": new_content,
            })
            draft.refinement_history = history
            draft.current_draft = new_content
            db.commit()
            db.refresh(draft)
            return draft
        return None
    finally:
        db.close()


def _approve_draft_sync(draft_id: str):
    """Approve a draft and add it to the main chat."""
    db = next(get_db())
    try:
        draft = db.get(DraftResponse, uuid.UUID(draft_id))
        if draft and draft.status == "active":
            # Save original user message
            user_message = ChatMessage(
                session_id=draft.session_id,
                role="user",
                content=draft.original_query
            )
            db.add(user_message)
            
            # Save approved AI response
            assistant_message = ChatMessage(
                session_id=draft.session_id,
                role="assistant",
                content=draft.current_draft
            )
            db.add(assistant_message)
            
            # Update session title if needed
            session = db.get(ChatSession, draft.session_id)
            if session and not session.title:
                session.title = draft.original_query[:50] + ("..." if len(draft.original_query) > 50 else "")
            
            # Mark draft as approved
            draft.status = "approved"
            db.commit()
            
            return {
                "user_message_id": str(user_message.id),
                "assistant_message_id": str(assistant_message.id),
                "draft_id": str(draft.id)
            }
        return None
    finally:
        db.close()


def _discard_draft_sync(draft_id: str):
    """Discard a draft."""
    db = next(get_db())
    try:
        draft = db.get(DraftResponse, uuid.UUID(draft_id))
        if draft and draft.status == "active":
            draft.status = "discarded"
            db.commit()
            return True
        return False
    finally:
        db.close()


def _refine_with_rag_sync(
    draft_id: str,
    refinement_request: str,
    top_k: int,
    session_id: str,
    history_k: int,
    channel: str,
    similarity_threshold: float = None,
):
    """Refine a draft using RAG to generate updated content."""
    db = next(get_db())
    try:
        draft = db.get(DraftResponse, uuid.UUID(draft_id))
        if not draft or draft.status != "active":
            return None
        
        rag_service = RAGService(db)
        
        # Build context from draft history
        draft_context = f"""
            Original Question: {draft.original_query}

            Current Draft Response:
            {draft.current_draft}

            User's Refinement Request: {refinement_request}

            Please update the draft response according to the user's refinement request. 
            Maintain the same style and format but incorporate the requested changes.
        """
        
        # Get chat history for context
        chat_history = _get_chat_history_sync(session_id, history_k)
        
        # Add draft refinement history to context
        if draft.refinement_history:
            for item in draft.refinement_history:
                chat_history.append(item)
        
        result = rag_service.query(
            query_text=draft_context,
            top_k=top_k,
            chat_history=chat_history,
            channel=channel,
            similarity_threshold=similarity_threshold,
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
    """Query API endpoint with chat history support and co-pilot mode."""
    try:
        body = await request.json()
        query_text = body.get("query", "").strip()
        top_k = body.get("top_k")
        session_id = body.get("session_id")
        history_k = body.get("history_k")  # Number of previous messages to include
        channel = (body.get("channel") or "email").lower()
        similarity_threshold = body.get("similarity_threshold")
        copilot_mode = body.get("copilot_mode", False)  # Co-pilot mode flag
        
        logger.info(
            f"Query API request: query='{query_text[:50]}...', session_id={session_id}, top_k={top_k}, history_k={history_k}, channel={channel}, copilot_mode={copilot_mode}"
        )

        if not query_text:
            logger.warning("Query API request missing query text")
            return JSONResponse({"error": "Query is required"}, status_code=400)

        # Execute query with history
        logger.debug("Executing query in thread pool...")
        result = await asyncio.to_thread(
            _query_with_history_sync,
            query_text,
            top_k,
            session_id,
            history_k,
            channel,
            similarity_threshold,
        )

        # Handle based on co-pilot mode
        if copilot_mode and session_id:
            # Create draft instead of saving to main chat
            sources_data = {
                "sources": result.get("sources", []),
                "kbs_used": result.get("kbs_used", []),
                "chunks": result.get("chunks", [])
            }
            draft = await asyncio.to_thread(
                _create_draft_sync,
                session_id,
                query_text,
                result["answer"],
                sources_data
            )
            result["draft_id"] = str(draft.id)
            result["is_draft"] = True
            logger.info(f"Created draft response: {draft.id}")
        elif session_id:
            # Normal mode - save directly to chat
            await asyncio.to_thread(_save_message_sync, session_id, "user", query_text)
            await asyncio.to_thread(
                _save_message_sync, session_id, "assistant", result["answer"]
            )
            result["is_draft"] = False
            logger.info(f"Saved messages to session: {session_id}")

        logger.info(f"Query completed successfully")
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ==================== Co-pilot Mode API Endpoints ====================

async def toggle_copilot_mode(request: Request) -> JSONResponse:
    """Toggle co-pilot mode for a session."""
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse({"error": "Session ID required"}, status_code=400)
    
    try:
        body = await request.json()
        enabled = body.get("enabled", False)
        
        success = await asyncio.to_thread(_toggle_copilot_mode_sync, session_id, enabled)
        if success:
            logger.info(f"Co-pilot mode {'enabled' if enabled else 'disabled'} for session: {session_id}")
            return JSONResponse({"success": True, "copilot_enabled": enabled})
        else:
            return JSONResponse({"error": "Session not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error toggling co-pilot mode: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_active_draft(request: Request) -> JSONResponse:
    """Get the active draft for a session."""
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse({"error": "Session ID required"}, status_code=400)
    
    try:
        draft = await asyncio.to_thread(_get_active_draft_sync, session_id)
        if draft:
            return JSONResponse({
                "draft_id": str(draft.id),
                "original_query": draft.original_query,
                "current_draft": draft.current_draft,
                "refinement_history": draft.refinement_history or [],
                "sources_data": draft.sources_data or {},
                "created_at": draft.created_at.isoformat(),
                "updated_at": draft.updated_at.isoformat()
            })
        else:
            return JSONResponse({"draft": None})
    except Exception as e:
        logger.error(f"Error getting active draft: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def refine_draft(request: Request) -> JSONResponse:
    """Refine a draft with a new request."""
    draft_id = request.path_params.get("draft_id")
    if not draft_id:
        return JSONResponse({"error": "Draft ID required"}, status_code=400)
    
    try:
        body = await request.json()
        refinement_request = body.get("refinement", "").strip()
        session_id = body.get("session_id")
        top_k = body.get("top_k", 7)
        history_k = body.get("history_k", 20)
        channel = (body.get("channel") or "email").lower()
        similarity_threshold = body.get("similarity_threshold")
        
        if not refinement_request:
            return JSONResponse({"error": "Refinement request is required"}, status_code=400)
        
        # Use RAG to generate refined content
        result = await asyncio.to_thread(
            _refine_with_rag_sync,
            draft_id,
            refinement_request,
            top_k,
            session_id,
            history_k,
            channel,
            similarity_threshold
        )
        
        if result is None:
            return JSONResponse({"error": "Draft not found or not active"}, status_code=404)
        
        # Update the draft with refined content
        updated_draft = await asyncio.to_thread(
            _refine_draft_sync,
            draft_id,
            refinement_request,
            result["answer"]
        )
        
        if updated_draft:
            logger.info(f"Refined draft: {draft_id}")
            return JSONResponse({
                "success": True,
                "draft_id": str(updated_draft.id),
                "current_draft": updated_draft.current_draft,
                "refinement_history": updated_draft.refinement_history or [],
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "kbs_used": result.get("kbs_used", []),
                "chunks": result.get("chunks", [])
            })
        else:
            return JSONResponse({"error": "Failed to update draft"}, status_code=500)
    except Exception as e:
        logger.error(f"Error refining draft: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def approve_draft(request: Request) -> JSONResponse:
    """Approve a draft and add it to the main chat."""
    draft_id = request.path_params.get("draft_id")
    if not draft_id:
        return JSONResponse({"error": "Draft ID required"}, status_code=400)
    
    try:
        result = await asyncio.to_thread(_approve_draft_sync, draft_id)
        if result:
            logger.info(f"Approved draft: {draft_id}")
            return JSONResponse({
                "success": True,
                **result
            })
        else:
            return JSONResponse({"error": "Draft not found or not active"}, status_code=404)
    except Exception as e:
        logger.error(f"Error approving draft: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def discard_draft(request: Request) -> JSONResponse:
    """Discard a draft."""
    draft_id = request.path_params.get("draft_id")
    if not draft_id:
        return JSONResponse({"error": "Draft ID required"}, status_code=400)
    
    try:
        success = await asyncio.to_thread(_discard_draft_sync, draft_id)
        if success:
            logger.info(f"Discarded draft: {draft_id}")
            return JSONResponse({"success": True})
        else:
            return JSONResponse({"error": "Draft not found or not active"}, status_code=404)
    except Exception as e:
        logger.error(f"Error discarding draft: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


chat_routes = [
    Route("/chat", chat_page, methods=["GET"]),
    Route("/api/chat/sessions", list_sessions, methods=["GET"]),
    Route("/api/chat/sessions", create_session, methods=["POST"]),
    Route("/api/chat/sessions/{session_id}", get_session, methods=["GET"]),
    Route("/api/chat/sessions/{session_id}", delete_session, methods=["DELETE"]),
    Route("/api/query", query_api, methods=["POST"]),
    # Co-pilot mode routes
    Route("/api/chat/sessions/{session_id}/copilot", toggle_copilot_mode, methods=["POST"]),
    Route("/api/chat/sessions/{session_id}/draft", get_active_draft, methods=["GET"]),
    Route("/api/draft/{draft_id}/refine", refine_draft, methods=["POST"]),
    Route("/api/draft/{draft_id}/approve", approve_draft, methods=["POST"]),
    Route("/api/draft/{draft_id}/discard", discard_draft, methods=["POST"]),
]
