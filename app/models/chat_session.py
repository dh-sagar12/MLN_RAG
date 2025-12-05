"""Chat Session model."""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from app.database import Base


class ChatSession(Base):
    """Chat Session model."""
    
    __tablename__ = "chat_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=True)  # Auto-generated from first message
    copilot_enabled = Column(Boolean, default=True, nullable=False)  # Co-pilot mode enabled by default
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")
    drafts = relationship("DraftResponse", back_populates="session", cascade="all, delete-orphan", order_by="DraftResponse.created_at.desc()")
    
    def __repr__(self) -> str:
        return f"<ChatSession(id={self.id}, title={self.title})>"


class ChatMessage(Base):
    """Chat Message model."""
    
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    
    def __repr__(self) -> str:
        return f"<ChatMessage(id={self.id}, role={self.role})>"


class DraftResponse(Base):
    """Draft Response model for Co-pilot mode."""
    
    __tablename__ = "draft_responses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    original_query = Column(Text, nullable=False)  # The user's original question
    current_draft = Column(Text, nullable=False)  # The current draft content
    status = Column(String(20), default="active", nullable=False)  # 'active', 'approved', 'discarded'
    refinement_history = Column(JSONB, default=list, nullable=False)  # List of refinement messages
    sources_data = Column(JSONB, default=dict, nullable=True)  # Store sources/chunks for reference
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    session = relationship("ChatSession", back_populates="drafts")
    
    def __repr__(self) -> str:
        return f"<DraftResponse(id={self.id}, status={self.status})>"

