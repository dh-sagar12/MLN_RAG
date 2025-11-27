"""Web Crawl Source model."""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


class WebCrawlSource(Base):
    """Web Crawl Source model for storing URLs to crawl."""
    
    __tablename__ = "web_crawl_sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kb_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    url = Column(Text, nullable=False)
    title = Column(String(500), nullable=True)  # Extracted title from the page
    description = Column(Text, nullable=True)  # Optional description/notes
    is_indexed = Column(Boolean, default=False, nullable=False)  # Whether content has been indexed
    last_crawled_at = Column(DateTime, nullable=True)  # Last successful crawl time
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="web_crawl_sources")
    
    def __repr__(self) -> str:
        return f"<WebCrawlSource(id={self.id}, url={self.url[:50]}...)>"

