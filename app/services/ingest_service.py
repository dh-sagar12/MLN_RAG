"""Ingestion service for processing documents and creating embeddings."""

import datetime
import logging
import asyncio
from typing import List, Dict, Any, Optional
from llama_index.core.schema import BaseNode
from sqlalchemy.orm import Session
from sqlalchemy import select
from llama_index.core import Document, Settings
from llama_index.core.node_parser import (
    HTMLNodeParser,
    SentenceSplitter,
    MarkdownNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from app.config import settings
from app.models import (
    KnowledgeBase,
    Embedding as EmbeddingModel,
    UploadedFile,
    WebCrawlSource,
)
from app.services.file_service import FileService
from app.database import SessionLocal
import uuid
from pgvector.sqlalchemy import Vector

logger = logging.getLogger(__name__)


class IngestService:
    """Service for ingesting documents and creating embeddings."""

    def __init__(self, db: Session):
        self.db = db
        self.file_service = FileService()

        # Initialize LlamaIndex settings
        if settings.openai_api_key:
            # Clear proxy environment variables to avoid OpenAI client initialization issues
            import os

            proxy_vars = [
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "http_proxy",
                "https_proxy",
                "ALL_PROXY",
                "all_proxy",
            ]
            saved_proxies = {}
            for var in proxy_vars:
                if var in os.environ:
                    saved_proxies[var] = os.environ.pop(var)

            try:
                Settings.embed_model = OpenAIEmbedding(
                    model=settings.openai_embedding_model,
                    api_key=settings.openai_api_key,
                )
                logger.info(
                    f"OpenAIEmbedding initialized with model: {settings.openai_embedding_model}"
                )
            except Exception as e:
                logger.error(f"Error initializing OpenAIEmbedding: {e}", exc_info=True)
                raise
            finally:
                # Restore proxy environment variables
                for var, value in saved_proxies.items():
                    os.environ[var] = value

        # Node parser
        self.node_parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200,
        )
        self.markdown_node_parser = MarkdownNodeParser()

    async def process_file(
        self, kb_id: uuid.UUID, file_path: str, file_type: str
    ) -> None:
        """Process a file and create embeddings.

        Args:
            kb_id: Knowledge base ID
            file_path: Path to the file
            file_type: File type
        """
        logger.info(f"Processing file: {file_path} (KB: {kb_id}, Type: {file_type})")

        try:
            # Extract text (async file I/O)
            logger.debug(f"Extracting text from file: {file_path}")
            text = await self.file_service.extract_text(file_path, file_type)
            logger.info(f"Extracted {len(text)} characters from file: {file_path}")

            # Create LlamaIndex document
            doc = Document(
                text=text,
                metadata={
                    "kb_id": str(kb_id),
                    "file_path": file_path,
                    "file_type": file_type,
                },
            )

            # Split into chunks
            logger.debug(f"Splitting document into chunks")
            nodes = self.node_parser.get_nodes_from_documents([doc])
            logger.info(f"Created {len(nodes)} chunks from file: {file_path}")

            # Create embeddings and store in database
            if settings.openai_api_key:
                embed_model = Settings.embed_model
                embeddings_data = []

                # Get all embeddings (run in thread pool)
                logger.info(f"Generating embeddings for {len(nodes)} chunks")
                for idx, node in enumerate(nodes):
                    logger.debug(
                        f"Generating embedding for chunk {idx + 1}/{len(nodes)}"
                    )
                    embedding_vector = await asyncio.to_thread(
                        embed_model.get_text_embedding, node.get_content()
                    )
                    embeddings_data.append(
                        {
                            "kb_id": kb_id,
                            "chunk_text": node.get_content(),
                            "embedding": embedding_vector,
                            "chunk_metadata": {
                                "node_id": node.node_id,
                                "file_path": file_path,
                                "file_type": file_type,
                                **node.metadata,
                            },
                        }
                    )

                # Save all embeddings in one batch (create new session for thread safety)
                if embeddings_data:
                    logger.info(f"Saving {len(embeddings_data)} embeddings to database")
                    await asyncio.to_thread(
                        self._save_embeddings_batch, embeddings_data
                    )
                    logger.info(f"Successfully saved embeddings for file: {file_path}")
            else:
                logger.warning(
                    "OpenAI API key not configured, skipping embedding generation"
                )
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            raise

    async def process_markdown(
        self,
        kb_id: uuid.UUID,
        markdown: str,
        source: Optional[WebCrawlSource] = None,
    ) -> None:
        """Process HTML and create embeddings.

        Args:
            kb_id: Knowledge base ID
            html: HTML content
            file_type: File type
        """
        logger.info(f"Processing Markdown: {markdown[:100]} (KB: {kb_id}\n")

        try:
            # Create LlamaIndex document
            doc = Document(
                text=markdown,
                metadata={
                    "kb_id": str(kb_id),
                    "url": source.url if source else None,
                    "source_type": "web_crawl",
                    "source_id": str(source.id) if source else None,
                    "title": source.title if source else None,
                    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
                },
            )
            # Split into chunks
            logger.debug(f"Splitting document into chunks")
            nodes = self.markdown_node_parser.get_nodes_from_documents([doc])
            logger.info(
                f"Created {len(nodes)} chunks from Markdown: {markdown[:100]}\n"
            )

            if settings.openai_api_key:
                embed_model = Settings.embed_model
                embeddings_data = []

                # Get all embeddings (run in thread pool)
                logger.info(f"Generating embeddings for {len(nodes)} chunks!")
                for idx, node in enumerate(nodes):
                    logger.debug(
                        f"Generating embedding for chunk {idx + 1}/{len(nodes)}"
                    )
                    with open("node.txt", "w") as f:
                        f.write(node.get_content())

                    embedding_vector = await asyncio.to_thread(
                        embed_model.get_text_embedding, node.get_content()
                    )
                    embeddings_data.append(
                        {
                            "kb_id": kb_id,
                            "chunk_text": node.get_content(),
                            "embedding": embedding_vector,
                            "chunk_metadata": {
                                "node_id": node.node_id,
                                "file_path": source.url if source else None,
                                "file_type": "markdown",
                                **node.metadata,
                            },
                        }
                    )

                # Save all embeddings in one batch (create new session for thread safety)
                if embeddings_data:
                    logger.info(f"Saving {len(embeddings_data)} embeddings to database")
                    await asyncio.to_thread(
                        self._save_embeddings_batch, embeddings_data
                    )
                    logger.info(
                        f"Successfully saved embeddings for Source: {source.url}"
                    )
            else:
                logger.warning(
                    "OpenAI API key not configured, skipping embedding generation"
                )

        except Exception as e:

            logger.error(
                f"Error processing Markdown: {markdown[:100]}: {str(e)}", exc_info=True
            )
            raise

    def _save_embeddings_batch(self, embeddings_data: List[Dict]) -> None:
        """Sync function to save multiple embeddings to database."""
        logger.debug(f"Saving batch of {len(embeddings_data)} embeddings to database")
        # Create new session for thread safety
        try:
            for emb_data in embeddings_data:
                embedding = EmbeddingModel(
                    kb_id=emb_data["kb_id"],
                    chunk_text=emb_data["chunk_text"],
                    embedding=emb_data["embedding"],
                    chunk_metadata=emb_data["chunk_metadata"],
                )
                self.db.add(embedding)
            self.db.commit()
            logger.debug(f"Successfully committed {len(embeddings_data)} embeddings")
        except Exception as e:
            logger.error(f"Error saving embeddings batch: {str(e)}", exc_info=True)
            self.db.rollback()
            raise

    def get_vector_store(self) -> PGVectorStore:
        """Get or create PGVectorStore."""
        # Create vector store connection
        # Note: LlamaIndex PGVectorStore uses sync SQLAlchemy, so we need to convert
        # For now, we'll use direct pgvector queries instead
        # This is a simplified approach
        pass

    async def create_embeddings_for_kb(self, kb_id: uuid.UUID) -> None:
        """Create embeddings for all files in a knowledge base."""
        logger.info(f"Creating embeddings for all files in KB: {kb_id}")

        # Get all files for this KB (sync operation, run in thread pool)
        files = await asyncio.to_thread(self._get_files_for_kb, kb_id)
        logger.info(f"Found {len(files)} files to process in KB: {kb_id}")

        # Process each file
        for idx, file in enumerate(files, 1):
            logger.info(f"Processing file {idx}/{len(files)}: {file.file_name}")
            try:
                await self.process_file(kb_id, file.file_path, file.file_type)
                logger.info(
                    f"Completed processing file {idx}/{len(files)}: {file.file_name}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to process file {file.file_name}: {str(e)}", exc_info=True
                )
                # Continue with next file
                continue

        logger.info(f"Finished processing all files for KB: {kb_id}")

    def _get_files_for_kb(self, kb_id: uuid.UUID):
        """Sync function to get files for KB."""
        result = self.db.execute(
            select(UploadedFile).where(UploadedFile.kb_id == kb_id)
        )
        return result.scalars().all()


# import uuid
# import logging
# import asyncio
# from typing import List

# from sqlalchemy.orm import Session
# from sqlalchemy import select

# from app.config import settings
# from app.models import UploadedFile

# from app.services.file_service import FileService

# # LlamaIndex imports
# from llama_index.core import Settings, SimpleDirectoryReader, Document
# from llama_index.core.ingestion import IngestionPipeline
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.llms.openai import OpenAI
# from llama_index.vector_stores.postgres import PGVectorStore
# from llama_index.core import VectorStoreIndex
# from llama_index.core.storage import StorageContext


# logger = logging.getLogger(__name__)


# class IngestService:
#     """Refactored ingestion pipeline using LlamaIndex built-in components."""

#     def __init__(self, db: Session):
#         self.db = db
#         self.file_service = FileService()

#         # Configure embedding model
#         if settings.openai_api_key:
#             Settings.embed_model = OpenAIEmbedding(
#                 model=settings.openai_embedding_model, api_key=settings.openai_api_key
#             )
#             Settings.llm = OpenAI(
#                 model=settings.openai_llm_model,
#                 api_key=settings.openai_api_key,
#                 temperature=settings.temperature,
#             )

#         # Built-in node parser (LlamaIndex default)
#         self.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

#     # ----------------------------------------------------------------------
#     # PGVectorStore initialization
#     # ----------------------------------------------------------------------
#     def _init_vector_store(self) -> PGVectorStore:
#         """Create PGVectorStore connection."""
#         return PGVectorStore.from_params(
#             database=settings.db_name,
#             host=settings.db_host,
#             password=settings.db_password,
#             port=settings.db_port,
#             user=settings.db_user,
#             table_name="embeddings",
#             embed_dim=settings.vector_dimension,
#         )

#     # ----------------------------------------------------------------------
#     # Main pipeline: file → LlamaIndex docs → vector store
#     # ----------------------------------------------------------------------
#     async def process_file(
#         self,
#         kb_id: uuid.UUID,
#         file_path: str,
#         file_type: str,
#     ):
#         logger.info(f"[Ingest] Processing file: {file_path} (KB: {kb_id})")

#         # 1. Load documents
#         reader = SimpleDirectoryReader(input_files=[file_path], recursive=False)
#         documents = reader.load_data()

#         for doc in documents:
#             doc.metadata.update(
#                 {"kb_id": str(kb_id), "file_path": file_path, "file_type": file_type}
#             )

#         # 2. Setup PGVectorStore + Storage
#         vector_store = self._init_vector_store()
#         storage_context = StorageContext.from_defaults(vector_store=vector_store)

#         # 3. Build async pipeline
#         pipeline = IngestionPipeline(
#             transformations=[self.node_parser],  # ONLY node parsing
#             vector_store=vector_store,
#         )


#         # 4. RUN ASYNC
#         nodes = await pipeline.arun(documents=documents)

#         logger.info(f"[Ingest] Generated {len(nodes)} nodes")

#         # 5. Build index + persist
#         index = VectorStoreIndex(nodes, storage_context=storage_context)

#         storage_context.persist()

#         logger.info(
#             f"[Ingest] Stored {len(nodes)} embeddings in PGVector for: {file_path}"
#         )

#     # ----------------------------------------------------------------------
#     # Process all files for a KB
#     # ----------------------------------------------------------------------
#     async def create_embeddings_for_kb(self, kb_id: uuid.UUID) -> None:
#         logger.info(f"[Ingest] Creating embeddings for KB: {kb_id}")

#         files = await asyncio.to_thread(self._get_files_for_kb, kb_id)

#         for idx, file in enumerate(files, 1):
#             try:
#                 logger.info(f"[Ingest] File {idx}/{len(files)} → {file.file_name}")
#                 await self.process_file(
#                     kb_id=kb_id, file_path=file.file_path, file_type=file.file_type
#                 )
#             except Exception as e:
#                 logger.error(
#                     f"[Ingest] Failed file {file.file_name}: {e}", exc_info=True
#                 )

#         logger.info(f"[Ingest] Completed ingestion for KB: {kb_id}")

#     # ----------------------------------------------------------------------
#     def _get_files_for_kb(self, kb_id: uuid.UUID):
#         """Get all uploaded files belonging to a KB."""
#         result = self.db.execute(
#             select(UploadedFile).where(UploadedFile.kb_id == kb_id)
#         )
#         return result.scalars().all()
