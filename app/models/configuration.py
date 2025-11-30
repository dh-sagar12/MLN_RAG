"""Configuration model for storing dynamic RAG settings."""

from sqlalchemy import Column, String, Float, Integer, Boolean, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
from app.database import Base


class Configuration(Base):
    """Configuration model for storing dynamic RAG settings."""
    
    __tablename__ = "configurations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=True)  # JSON string for complex values
    value_type = Column(String(20), nullable=False, default="string")  # string, int, float, bool, json
    description = Column(Text, nullable=True)  # Help text for UI
    category = Column(String(50), nullable=False, index=True)  # rag, retriever, prompt, etc.
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<Configuration(key='{self.key}', value='{self.value}', category='{self.category}')>"
    
    def get_typed_value(self):
        """Convert value to appropriate type based on value_type."""
        if self.value is None:
            return None
        
        if self.value_type == "int":
            return int(self.value)
        elif self.value_type == "float":
            return float(self.value)
        elif self.value_type == "bool":
            return self.value.lower() in ("true", "1", "yes", "on")
        elif self.value_type == "json":
            import json
            return json.loads(self.value)
        else:
            return self.value
    
    def set_typed_value(self, value):
        """Set value with appropriate type conversion."""
        if value is None:
            self.value = None
            return
        
        if isinstance(value, bool):
            self.value_type = "bool"
            self.value = str(value).lower()
        elif isinstance(value, int):
            self.value_type = "int"
            self.value = str(value)
        elif isinstance(value, float):
            self.value_type = "float"
            self.value = str(value)
        elif isinstance(value, (dict, list)):
            self.value_type = "json"
            import json
            self.value = json.dumps(value)
        else:
            self.value_type = "string"
            self.value = str(value)

