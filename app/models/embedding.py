"""Embedding model."""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    Text,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.database import Base
from app.config import settings


class Embedding(Base):
    """Embedding model with pgvector support."""

    __tablename__ = "embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kb_id = Column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(settings.vector_dimension), nullable=False)
    chunk_metadata = Column(JSON, nullable=True, default=dict)
    chunk_text_tsv = Column(TSVECTOR, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="embeddings")

    def __repr__(self) -> str:
        return f"<Embedding(id={self.id}, kb_id={self.kb_id})>"
