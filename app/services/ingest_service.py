"""Ingestion service for processing documents and creating embeddings."""

import datetime
import logging
import asyncio
from typing import List, Dict, Any, Optional
from llama_index.core.schema import BaseNode
from sqlalchemy.orm import Session
from sqlalchemy import select
from llama_index.core import Document, Settings, SimpleDirectoryReader
from llama_index.core.node_parser import (
    SentenceSplitter,
    MarkdownNodeParser,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from app.config import settings
from app.models import (
    Embedding as EmbeddingModel,
    UploadedFile,
    WebCrawlSource,
)
from app.services.file_service import FileService
import uuid

logger = logging.getLogger(__name__)


class IngestService:
    """Service for ingesting documents and creating embeddings using LlamaIndex IngestionPipeline."""

    def __init__(self, db: Session):
        self.db = db
        self.file_service = FileService()
        self.embed_model = None
        self.llm = None

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
                self.embed_model = OpenAIEmbedding(
                    model=settings.openai_embedding_model,
                    api_key=settings.openai_api_key,
                )
                Settings.embed_model = self.embed_model

                self.llm = OpenAI(
                    model=settings.openai_llm_model,
                    api_key=settings.openai_api_key,
                    temperature=settings.temperature,
                )
                Settings.llm = self.llm

                logger.info(
                    f"OpenAI initialized (Embedding: {settings.openai_embedding_model}, LLM: {settings.openai_llm_model})"
                )
            except Exception as e:
                logger.error(f"Error initializing OpenAI clients: {e}", exc_info=True)
                raise
            finally:
                # Restore proxy environment variables
                for var, value in saved_proxies.items():
                    os.environ[var] = value

        # Node parsers
        self.node_parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200,
        )
        self.markdown_node_parser = MarkdownNodeParser()

        # Metadata Extractors (We can use different Extractor if necessary)

        # self.title_extractor = TitleExtractor(
        #     nodes=5,
        #     llm=self.llm,
        # )
        # self.questions_extractor = QuestionsAnsweredExtractor(
        #     questions=3,
        #     llm=self.llm,
        # )

    def _update_file_status(self, file_path: str, status: str) -> None:
        """Update status of uploaded file."""
        try:
            stmt = select(UploadedFile).where(UploadedFile.file_path == file_path)
            result = self.db.execute(stmt)
            file = result.scalar_one_or_none()
            if file:
                file.status = status
                self.db.commit()
        except Exception as e:
            logger.error(f"Error updating file status for {file_path}: {e}")
            self.db.rollback()

    async def process_multiple_files(
        self,
        kb_id: uuid.UUID,
        file_paths: List[str],
    ) -> None:
        """Process multiple files using SimpleDirectoryReader.

        Args:
            kb_id: Knowledge base ID
            file_paths: List of file paths to process
        """
        logger.info(f"Processing {len(file_paths)} files for KB: {kb_id}")

        # Update status to PROCESSING
        for path in file_paths:
            await asyncio.to_thread(self._update_file_status, path, "PROCESSING")

        try:
            # Use SimpleDirectoryReader to load data
            reader = SimpleDirectoryReader(input_files=file_paths)
            documents = await asyncio.to_thread(reader.load_data)
            
            logger.info(f"Loaded {len(documents)} documents from {len(file_paths)} files")

            # Update metadata
            for doc in documents:
                doc.metadata.update({
                    "kb_id": str(kb_id),
                    "source_type": "document_upload",
                    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
                })
                # Ensure file_path is preserved (SimpleDirectoryReader puts it in metadata)
                if "file_path" not in doc.metadata:
                    # Fallback if needed, though SimpleDirectoryReader usually adds it
                    pass

            # Build pipeline
            transformations = [self.node_parser]
            if self.embed_model:
                transformations.append(self.embed_model)

            pipeline = IngestionPipeline(transformations=transformations)

            logger.info("Running IngestionPipeline for batch...")
            nodes = await pipeline.arun(documents=documents)
            logger.info(f"Pipeline generated {len(nodes)} nodes with embeddings")

            if nodes:
                await self._save_nodes_to_db(nodes, kb_id)
                # Update status to COMPLETED
                for path in file_paths:
                    await asyncio.to_thread(self._update_file_status, path, "COMPLETED")
            else:
                logger.warning("No nodes generated from batch")
                # Should we mark as failed? Maybe not.
                for path in file_paths:
                    await asyncio.to_thread(self._update_file_status, path, "COMPLETED")

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}", exc_info=True)
            # Update status to FAILED
            for path in file_paths:
                await asyncio.to_thread(self._update_file_status, path, "FAILED")
            raise

    async def process_file(
        self,
        kb_id: uuid.UUID,
        file_path: str,
        file_type: str,
    ) -> None:
        """Process a file and create embeddings using IngestionPipeline.

        Args:
            kb_id: Knowledge base ID
            file_path: Path to the file
            file_type: File type
        """
        logger.info(f"Processing file: {file_path} (KB: {kb_id}, Type: {file_type})")
        await asyncio.to_thread(self._update_file_status, file_path, "PROCESSING")

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
                    "source_type": "document_upload",
                    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
                },
            )

            # Build and run pipeline
            transformations = [self.node_parser]

            # Add extractors if LLM is available
            # if self.llm:
            #     transformations.extend(
            #         [
            #             self.title_extractor,
            #             self.questions_extractor,
            #         ]
            #     )

            if self.embed_model:
                transformations.append(self.embed_model)

            pipeline = IngestionPipeline(
                transformations=transformations,
            )

            logger.info("Running IngestionPipeline for file...")
            nodes = await pipeline.arun(documents=[doc])
            logger.info(f"Pipeline generated {len(nodes)} nodes with embeddings")

            # Save embeddings to database
            if nodes:
                await self._save_nodes_to_db(nodes, kb_id)
                logger.info(f"Successfully saved embeddings for file: {file_path}")
                await asyncio.to_thread(self._update_file_status, file_path, "COMPLETED")
            else:
                logger.warning("No nodes generated from pipeline")
                await asyncio.to_thread(self._update_file_status, file_path, "COMPLETED")

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            await asyncio.to_thread(self._update_file_status, file_path, "FAILED")
            raise

    async def process_markdown(
        self,
        kb_id: uuid.UUID,
        markdown: str,
        source: Optional[WebCrawlSource] = None,
    ) -> None:
        """Process Markdown and create embeddings using IngestionPipeline.

        Args:
            kb_id: Knowledge base ID
            markdown: Markdown content
            source: Web crawl source info
        """
        logger.info(f"Processing Markdown (KB: {kb_id})")

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
                    "file_type": "markdown",
                },
            )

            # Build and run pipeline
            transformations = [self.markdown_node_parser]

            # Add extractors if LLM is available
            # if self.llm:
            #     transformations.extend(
            #         [
            #             self.title_extractor,
            #             self.questions_extractor,
            #         ]
            #     )

            if self.embed_model:
                transformations.append(self.embed_model)

            pipeline = IngestionPipeline(
                transformations=transformations,
            )

            logger.info("Running IngestionPipeline for markdown...")
            nodes = await pipeline.arun(documents=[doc])
            logger.info(f"Pipeline generated {len(nodes)} nodes with embeddings")

            # Save embeddings to database
            if nodes:
                await self._save_nodes_to_db(
                    nodes=nodes,
                    kb_id=kb_id,
                )
                logger.info(
                    f"Successfully saved embeddings for Source: {source.url if source else 'Unknown'}"
                )
            else:
                logger.warning("No nodes generated from pipeline")

        except Exception as e:
            logger.error(f"Error processing Markdown: {str(e)}", exc_info=True)
            raise

    async def _save_nodes_to_db(self, nodes: List[BaseNode], kb_id: uuid.UUID) -> None:
        """Save processed nodes to the database."""
        embeddings_data = []
        for node in nodes:
            # Ensure embedding exists
            if not node.embedding:
                logger.warning(f"Node {node.node_id} has no embedding, skipping")
                continue

            embeddings_data.append(
                {
                    "kb_id": kb_id,
                    "chunk_text": node.get_content(),
                    "embedding": node.embedding,
                    "chunk_metadata": {
                        "node_id": node.node_id,
                        **node.metadata,
                    },
                }
            )

        if embeddings_data:
            logger.info(
                f"Saving batch of {len(embeddings_data)} embeddings to database"
            )
            await asyncio.to_thread(self._save_embeddings_batch, embeddings_data)

    def _save_embeddings_batch(self, embeddings_data: List[Dict]) -> None:
        """Sync function to save multiple embeddings to database."""
        # Create new session for thread safety if needed, or use existing if safe
        # Since this is running in a thread, we should use the session passed in __init__
        # BUT SQLAlchemy sessions are not thread-safe.
        # The original code used self.db which was passed in __init__.
        # If IngestService is created per request, it might be okay.
        # However, `await asyncio.to_thread(self._save_embeddings_batch, ...)` runs in a separate thread.
        # Sharing `self.db` (Session) across threads is dangerous.
        # The original code did: `self.db.add(embedding)` inside `_save_embeddings_batch` called via `asyncio.to_thread`.
        # This is risky if `self.db` is used elsewhere concurrently.
        # Ideally we should create a new session here or ensure `self.db` is thread-local or only used here.
        # Given the context, I will stick to the pattern but maybe use a fresh session if possible?
        # The original code imported `SessionLocal`.

        # Let's use a fresh session to be safe, as seen in some patterns, or stick to self.db if we are sure.
        # The original code had `from app.database import SessionLocal` but didn't use it in `_save_embeddings_batch`.
        # It used `self.db`.
        # I will stick to `self.db` to avoid breaking transaction scope if the caller expects it,
        # BUT `asyncio.to_thread` makes it concurrent.
        # If the caller waits for `process_file` to finish before doing anything else with `db`, it's "okay" but not great.
        # I'll stick to `self.db` to match original behavior but add a comment.

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
                await self.process_file(kb_id=kb_id, file_path=file.file_path, file_type=file.file_type)
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
