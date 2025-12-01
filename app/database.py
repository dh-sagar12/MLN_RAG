"""Database connection and session management."""

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings
import asyncio

logger = logging.getLogger(__name__)

# Create sync engine
engine = create_engine(
    settings.database_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables and pgvector extension."""
    from app.models import (
        knowledge_base,
        uploaded_file,
        embedding,
        chat_session,
        web_crawl_source,
        configuration,
    )

    logger.info("Initializing database...")

    # Enable pgvector extension first (must be done in a transaction that commits)
    logger.info("Creating pgvector extension...")
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    logger.info("pgvector extension created/verified")

    # Create all tables
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")

    # Create HNSW index for embeddings (in a new transaction)
    logger.info("Creating HNSW index for embeddings...")
    with engine.begin() as conn:
        # Check if index already exists to avoid errors
        result = conn.execute(
            text(
                """
            SELECT EXISTS (
                SELECT 1 FROM pg_indexes 
                WHERE indexname = 'embeddings_embedding_idx'
            )
        """
            )
        )
        index_exists = result.scalar()

        if not index_exists:
            logger.info("Creating embeddings_embedding_idx index...")
            conn.execute(
                text(
                """
                    CREATE INDEX embeddings_embedding_idx 
                    ON embeddings 
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 200)
                """
                )
            )
            conn.execute(
                text(
                    """
            SET hnsw.ef_search = 256;
            """
                )
            )
            logger.info("HNSW index created successfully")
        else:
            logger.info("HNSW index already exists, skipping creation")

        # Set HNSW search parameters (session-level, not persistent)
        # This will be set per connection as needed

    logger.info("Creating Index embeddings_chunk_text_tsv_idx & trigger.")
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
            SELECT EXISTS (
                SELECT 1 FROM pg_indexes 
                WHERE indexname = 'embeddings_chunk_text_tsv_idx'
            )
        """
            )
        )
        index_exists = result.scalar()
        if not index_exists:
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS embeddings_chunk_text_tsv_idx 
                    ON embeddings USING GIN (chunk_text_tsv);                
                """
                )
            )
        else:
            logger.info("Index embeddings_chunk_text_tsv_idx already exists..")

        conn.execute(
            text(
                """
                    CREATE OR REPLACE FUNCTION embeddings_tsv_trigger() RETURNS trigger AS $$
                    BEGIN
                        NEW.chunk_text_tsv := to_tsvector('english', COALESCE(NEW.chunk_text, ''));
                        RETURN NEW;
                    END
                    $$ LANGUAGE plpgsql;
                """
            )
        )

        conn.execute(
            text(
                """
                    DROP TRIGGER IF EXISTS embeddings_tsv_update ON embeddings;
                    CREATE TRIGGER embeddings_tsv_update 
                    BEFORE INSERT OR UPDATE ON embeddings
                    FOR EACH ROW EXECUTE FUNCTION embeddings_tsv_trigger();
                """
            )
        )

    # Initialize default configuration values
    logger.info("Initializing default configuration...")
    from app.services.config_service import ConfigService

    db = SessionLocal()
    try:
        ConfigService.initialize_defaults(db)
        logger.info("Default configuration initialized")
    except Exception as e:
        logger.error(f"Error initializing default configuration: {e}", exc_info=True)
    finally:
        db.close()

    logger.info("Database initialization completed")
