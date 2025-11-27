"""Knowledge Base model."""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


class KnowledgeBase(Base):
    """Knowledge Base model."""
    
    __tablename__ = "knowledge_bases"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    files = relationship("UploadedFile", back_populates="knowledge_base", cascade="all, delete-orphan")
    embeddings = relationship("Embedding", back_populates="knowledge_base", cascade="all, delete-orphan")
    web_crawl_sources = relationship("WebCrawlSource", back_populates="knowledge_base", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<KnowledgeBase(id={self.id}, name={self.name})>"

