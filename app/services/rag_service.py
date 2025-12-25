"""RAG service for querying across knowledge bases."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
)

from llama_index.core import (
    ChatPromptTemplate,
    QueryBundle,
    get_response_synthesizer,
)
from llama_index.core.llms import ChatMessage as LlamaChatMessage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.cohere_rerank import CohereRerank
from sqlalchemy.orm import Session

from app.config import settings
from app.models import DraftResponse
from app.services.config_service import ConfigService
from app.services.extractor import IntentMetadataFilter
from app.services.performance_tracker import PerformanceTracker, get_performance_tracker
from app.services.retriever import BM25Retriever, PostgresRetriever

if TYPE_CHECKING:
    from llama_index.core.base.response.schema import RESPONSE_TYPE


logger = logging.getLogger(__name__)


# ============================================================================
# Enums & Constants
# ============================================================================


class Channel(str, Enum):
    """Supported communication channels."""

    EMAIL = "email"
    WHATSAPP = "whatsapp"

    @classmethod
    def from_string(cls, value: str) -> "Channel":
        """Convert string to Channel enum, defaulting to EMAIL."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.EMAIL


class ResponseMode(str, Enum):
    """Response synthesis modes."""

    REFINE = "refine"
    COMPACT = "compact"
    TREE_SUMMARIZE = "tree_summarize"
    SIMPLE_SUMMARIZE = "simple_summarize"
    ACCUMULATE = "accumulate"
    COMPACT_ACCUMULATE = "compact_accumulate"
    GENERATION = "generation"
    NO_TEXT = "no_text"
    CONTEXT_ONLY = "context_only"


# ============================================================================
# Type Definitions
# ============================================================================


class ChunkData(TypedDict):
    """Type definition for retrieved chunk data."""

    text: str
    kb_id: Optional[str]
    kb_name: str
    similarity: float
    metadata: Dict[str, Any]


class QueryResult(TypedDict):
    """Type definition for query result."""

    answer: str
    sources: List[str]
    kbs_used: List[str]
    chunks: List[ChunkData]
    request_id: Optional[str]


class EnhancedQueryResult(TypedDict):
    """Type definition for enhanced query with intents."""

    enhanced_query: str
    intents: List[str]


@dataclass
class QueryContext:
    """Context holder for query execution."""

    query_text: str
    enhanced_query: str
    intents: List[str]
    top_k: int
    similarity_threshold: float
    channel: Channel
    chat_history: Optional[List[Dict[str, str]]]
    session_id: Optional[str]
    draft_mode: bool
    request_id: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from retrieval phase."""

    chunks: List[ChunkData] = field(default_factory=list)
    kbs_used: Set[str] = field(default_factory=set)
    sources: List[str] = field(default_factory=list)
    context_text: str = ""


# ============================================================================
# RAG Service Implementation
# ============================================================================


class RAGService:
    """Service for RAG querying using LlamaIndex.

    This service provides retrieval-augmented generation capabilities,
    supporting hybrid search (vector + BM25), query enhancement,
    and multi-channel response formatting.
    """

    # Proxy environment variables to manage during initialization
    _PROXY_ENV_VARS: Tuple[str, ...] = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    )

    def __init__(self, db: Session) -> None:
        """Initialize the RAG service.

        Args:
            db: SQLAlchemy database session.
        """
        self.db = db
        self.embed_model: Optional[OpenAIEmbedding] = None
        self.llm: Optional[OpenAI] = None
        self.filters: Optional[MetadataFilters] = None
        self.performance_tracker: Optional[PerformanceTracker] = None

        # Draft mode state (set during refine_draft)
        self._draft_mode_prompt_template: Optional[List[LlamaChatMessage]] = None
        self._enhancement_draft_history: Optional[List[LlamaChatMessage]] = None

        # Load configuration and initialize clients
        self._load_config()
        self._initialize_performance_tracker()
        self._initialize_openai_clients()

    @staticmethod
    def _build_draft_retrieval_hint(draft_text: str, *, max_chars: int = 900) -> str:
        """Build a compact retrieval hint from the current draft.

        Goal: keep retrieval grounded in previously-covered topics during refinement,
        without embedding the entire draft (which can be noisy and large).
        """
        if not draft_text:
            return ""

        lines = [ln.strip() for ln in draft_text.splitlines() if ln.strip()]
        kept: List[str] = []

        for ln in lines:
            # Prefer structure cues (headers/bullets/short lines)
            is_structured = (
                ln.startswith(("#", "-", "*"))
                or ln.startswith("**")
                or ln.lower().startswith(("day ", "itinerary", "weather", "food"))
            )
            if is_structured or len(ln) <= 140:
                kept.append(ln)

            # Stop once we have enough material
            if sum(len(x) + 1 for x in kept) >= max_chars:
                break

        hint = " | ".join(kept)
        return hint[:max_chars].strip()

    # ========================================================================
    # Initialization Methods
    # ========================================================================

    def _load_config(self) -> None:
        """Load dynamic configuration from database."""
        self.llm_config = ConfigService.get_llm_config(self.db)
        self.rag_config = ConfigService.get_rag_config(self.db)
        self.retriever_config = ConfigService.get_retriever_config(self.db)
        self.hybrid_config = ConfigService.get_hybrid_config(self.db)
        self.prompt_config = ConfigService.get_prompt_config(self.db)

    def _initialize_performance_tracker(self) -> None:
        """Initialize the performance tracker if enabled."""
        if settings.performance_tracking_enabled:
            self.performance_tracker = get_performance_tracker(
                log_dir=settings.performance_log_dir,
                log_filename=settings.performance_log_filename,
                max_file_size_mb=settings.performance_max_file_size_mb,
                flush_interval=settings.performance_flush_interval,
            )
            logger.info("Performance tracking enabled")

    def _initialize_openai_clients(self) -> None:
        """Initialize OpenAI embedding and LLM clients."""
        if not settings.openai_api_key:
            logger.warning("OpenAI API key not configured")
            return

        # Temporarily clear proxy environment variables
        saved_proxies = self._clear_proxy_env_vars()

        try:
            llm_model = self.llm_config.get("model")
            llm_temperature = self.llm_config.get("temperature")
            embedding_model = self.llm_config.get("embedding_model")

            self.embed_model = OpenAIEmbedding(
                model=embedding_model,
                api_key=settings.openai_api_key,
            )
            self.llm = OpenAI(
                model=llm_model,
                api_key=settings.openai_api_key,
                temperature=llm_temperature,
            )

            logger.info(
                f"OpenAI clients initialized "
                f"(embedding: {embedding_model}, LLM: {llm_model}, "
                f"temperature: {llm_temperature})"
            )
        except Exception as e:
            logger.error(f"Error initializing OpenAI clients: {e}", exc_info=True)
            raise
        finally:
            self._restore_proxy_env_vars(saved_proxies)

    def _clear_proxy_env_vars(self) -> Dict[str, str]:
        """Clear proxy environment variables and return saved values."""
        saved_proxies = {}
        for var in self._PROXY_ENV_VARS:
            if var in os.environ:
                saved_proxies[var] = os.environ.pop(var)
        return saved_proxies

    def _restore_proxy_env_vars(self, saved_proxies: Dict[str, str]) -> None:
        """Restore previously saved proxy environment variables."""
        for var, value in saved_proxies.items():
            os.environ[var] = value

    # ========================================================================
    # Query Enhancement
    # ========================================================================

    def _get_enhanced_query_and_intents(
        self,
        *,
        query_text: str,
        chat_history: Optional[List[Dict[str, str]]],
        draft_mode: bool,
    ) -> EnhancedQueryResult:
        """Generate enhanced query and extract intents using LLM.

        Args:
            query_text: The original user query.
            chat_history: Previous conversation messages.
            draft_mode: Whether this is a draft refinement request.

        Returns:
            EnhancedQueryResult with enhanced query and detected intents.
        """
        if not self.llm:
            return {
                "enhanced_query": query_text,
                "intents": [],
            }

        messages = self._build_enhancement_messages(
            query_text=query_text,
            chat_history=chat_history,
            draft_mode=draft_mode,
        )
        
        logger.info(f"Query enhancement messages: {messages}")

        response = self.llm.chat(messages)
        logger.debug(f"Query enhancement response: {response.message.content}")

        return self._parse_enhancement_response(
            response_content=response.message.content,
        )

    def _build_enhancement_messages(
        self,
        *,
        query_text: str,
        chat_history: Optional[List[Dict[str, str]]],
        draft_mode: bool,
    ) -> List[LlamaChatMessage]:
        """Build messages for query enhancement LLM call.

        Args:
            query_text: The original user query.
            chat_history: Previous conversation messages.
            draft_mode: Whether this is a draft refinement request.

        Returns:
            List of chat messages for the LLM.
        """
        system_prompt = self.prompt_config.get("query_enhancement")
        messages = [
            LlamaChatMessage(
                role="system",
                content=system_prompt,
            )
        ]

        # Add chat history context
        if chat_history and len(chat_history) > 0:
            messages.append(
                LlamaChatMessage(
                    role="developer",
                    content="Chat Histories (previous chats with the end customer):\n-------------------\n",
                )
            )
            for msg in chat_history:
                messages.append(
                    LlamaChatMessage(
                        role=msg["role"],
                        content=msg["content"],
                    )
                )

        # Add draft mode context if applicable
        if draft_mode and self._enhancement_draft_history:
            messages.extend(self._enhancement_draft_history or [])

        # Add the current query
        role = "developer" if draft_mode else "user"
        messages.append(
            LlamaChatMessage(
                role=role,
                content=query_text,
            )
        )

        return messages

    def _parse_enhancement_response(
        self,
        *,
        response_content: str,
    ) -> EnhancedQueryResult:
        """Parse the LLM response for query enhancement.

        Args:
            response_content: Raw LLM response content.
            fallback_query: Query to use if parsing fails.

        Returns:
            EnhancedQueryResult with enhanced query and intents.
        """
        try:
            decoded_response = json.loads(response_content)
            return {
                "enhanced_query": decoded_response["enhanced_query"],
                "intents": decoded_response.get("intents", []),
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing enhancement response: {e}", exc_info=True)
            return {"enhanced_query": str(response_content).strip(), "intents": []}

    # ========================================================================
    # Prompt Building
    # ========================================================================

    def get_system_prompt_for_channel(self, channel: Channel) -> str:
        """Return system prompt instructions tailored to a channel.

        Args:
            channel: The communication channel.

        Returns:
            Combined system prompt for the channel.
        """
        base_prompt = self.prompt_config.get("base")
        channel_prompts = {
            Channel.EMAIL: self.prompt_config.get("email"),
            Channel.WHATSAPP: self.prompt_config.get("whatsapp"),
        }

        channel_guidance = channel_prompts.get(channel, channel_prompts[Channel.EMAIL])

        if channel_guidance:
            return f"{base_prompt}\n\n{channel_guidance}"
        return base_prompt

    def _build_qa_template(
        self,
        *,
        channel: Channel,
        chat_history: Optional[List[Dict[str, str]]],
        original_query: str,
        intents: List[str],
        draft_mode: bool,
    ) -> ChatPromptTemplate:
        """Build the QA prompt template for response generation.

        Args:
            channel: Communication channel for response formatting.
            chat_history: Previous conversation messages.
            original_query: The user's original query.
            intents: Detected query intents.
            draft_mode: Whether this is a draft refinement.

        Returns:
            ChatPromptTemplate for the response synthesizer.
        """
        system_prompt = self.get_system_prompt_for_channel(channel)
        messages: List[LlamaChatMessage] = [
            LlamaChatMessage(
                role="system",
                content=system_prompt,
            )
        ]

        # Add draft mode template if applicable
        if draft_mode and self._draft_mode_prompt_template:
            messages.extend(self._draft_mode_prompt_template)

        # Add chat history
        if chat_history and len(chat_history) > 0:
            messages.append(
                LlamaChatMessage(
                    role="developer",
                    content="Chat Histories (previous chats with the end customer):\n-------------------\n",
                )
            )
            for msg in chat_history:
                messages.append(
                    LlamaChatMessage(
                        role=msg["role"],
                        content=msg["content"],
                    )
                )

        # Add context and query messages
        messages.extend(
            [
                LlamaChatMessage(
                    role="developer",
                    content="Context (retrieved context from the knowledge bases):\n---------------------\n{context_str}\n---------------------",
                ),
                LlamaChatMessage(
                    role="developer",
                    content=f"Intents: {intents}",
                ),
            ]
        )

        return ChatPromptTemplate(message_templates=messages)

    # ========================================================================
    # Retriever Configuration
    # ========================================================================

    def _create_metadata_filters(self, intents: List[str]) -> Optional[MetadataFilters]:
        """Create metadata filters based on intents.

        Args:
            intents: List of detected intents.

        Returns:
            MetadataFilters if intent filtering is enabled, None otherwise.
        """
        if not self.retriever_config["vector"]["use_intent_filter"]:
            return None

        return MetadataFilters(
            filters=[
                IntentMetadataFilter(
                    key="intent_categories",
                    value=intents,
                )
            ]
        )

    def _create_retrievers(
        self,
        *,
        similarity_threshold: float,
        metadata_filters: Optional[MetadataFilters],
        request_id: Optional[str],
    ) -> List[BaseRetriever]:
        """Create the list of retrievers for hybrid search.

        Args:
            similarity_threshold: Minimum similarity score for vector search.
            metadata_filters: Optional metadata filters.
            request_id: Request ID for performance tracking.

        Returns:
            List of configured retrievers.
        """
        retrievers: List[BaseRetriever] = []

        # Vector retriever (always included)
        vector_retriever = PostgresRetriever(
            db=self.db,
            embed_model=self.embed_model,
            similarity_threshold=similarity_threshold,
            performance_tracker=self.performance_tracker,
            request_id=request_id,
            metadata_filters=metadata_filters,
        )
        retrievers.append(vector_retriever)

        # BM25 retriever (optional)
        if self.retriever_config["bm25"]["enabled"]:
            bm25_retriever = BM25Retriever(
                db=self.db,
                performance_tracker=self.performance_tracker,
                request_id=request_id,
            )
            retrievers.append(bm25_retriever)

        return retrievers

    def _create_hybrid_retriever(
        self,
        *,
        retrievers: List[BaseRetriever],
        top_k: int,
    ) -> QueryFusionRetriever:
        """Create the hybrid fusion retriever.

        Args:
            retrievers: List of base retrievers to combine.
            top_k: Number of top results to return.

        Returns:
            Configured QueryFusionRetriever.
        """
        return QueryFusionRetriever(
            retrievers=retrievers,
            similarity_top_k=top_k,
            num_queries=self.hybrid_config["num_queries"],
            query_gen_prompt=self.prompt_config.get("query_enhancement"),
            mode=self.hybrid_config["mode"],
            use_async=False,
            llm=self.llm,
        )

    def _create_response_synthesizer(
        self,
        *,
        qa_template: ChatPromptTemplate,
    ) -> BaseSynthesizer:
        """Create the response synthesizer.

        Args:
            qa_template: The QA prompt template.

        Returns:
            Configured response synthesizer.
        """
        response_mode = self.rag_config["response_mode"]
        return get_response_synthesizer(
            llm=self.llm,
            text_qa_template=qa_template,
            response_mode=response_mode,
        )
        
    def _get_reranker_model(self, top_k: int) -> CohereRerank:
        """Get the reranker model.

        Returns:
            Configured reranker model.
        """
        return CohereRerank(
            top_n=top_k,
            api_key=settings.cohere_api_key,
            model="rerank-english-v3.0",
        )

    def _create_node_postprocessors(self, *, top_k: int) -> List[Any]:
        """Create node postprocessors for reranking.

        Args:
            top_k: Number of top results to keep after reranking.

        Returns:
            List of node postprocessors.
        """
        postprocessors = []

        if self.retriever_config["use_reranker"]:
            reranker = self._get_reranker_model(top_k=top_k)
            postprocessors.append(reranker)

        return postprocessors

    # ========================================================================
    # Result Processing
    # ========================================================================

    def _extract_retrieval_result(
        self,
        response: "RESPONSE_TYPE",
    ) -> RetrievalResult:
        """Extract structured results from the query response.

        Args:
            response: The LlamaIndex query response.

        Returns:
            RetrievalResult with chunks, sources, and metadata.
        """
        result = RetrievalResult()

        for node_with_score in response.source_nodes:
            node = node_with_score.node
            score = node_with_score.score

            kb_name = node.metadata.get("kb_name", "Unknown")
            chunk_text = node.get_content()
            result.context_text += chunk_text + "\n\n"

            chunk_data: ChunkData = {
                "text": chunk_text,
                "kb_id": node.metadata.get("kb_id"),
                "kb_name": kb_name,
                "similarity": float(score) if score is not None else None,
                "metadata": node.metadata,
            }
            result.chunks.append(chunk_data)
            result.kbs_used.add(kb_name)

            if "file_path" in node.metadata:
                result.sources.append(node.metadata["file_path"])

        return result

    def _build_query_result(
        self,
        *,
        answer_text: str,
        retrieval_result: RetrievalResult,
        request_id: Optional[str],
    ) -> QueryResult:
        """Build the final query result.

        Args:
            answer_text: Generated answer text.
            retrieval_result: Results from retrieval phase.
            request_id: Optional request ID for tracking.

        Returns:
            QueryResult dictionary.
        """
        result: QueryResult = {
            "answer": answer_text,
            "sources": list(set(retrieval_result.sources)),
            "kbs_used": list(retrieval_result.kbs_used),
            "chunks": retrieval_result.chunks,
            "request_id": request_id,
        }
        return result

    # ========================================================================
    # Performance Tracking
    # ========================================================================

    def _track_query_metrics(
        self,
        *,
        request_id: str,
        original_query: str,
        enhanced_query: str,
        channel: Channel,
        chat_history_length: int,
    ) -> None:
        """Record query-related performance metrics.

        Args:
            request_id: Request ID for tracking.
            original_query: Original user query.
            enhanced_query: Enhanced query after LLM processing.
            channel: Communication channel.
            chat_history_length: Number of messages in chat history.
        """
        if not self.performance_tracker or not request_id:
            return

        self.performance_tracker.record_query(
            request_id=request_id,
            original_query=original_query,
            enhanced_query=enhanced_query,
            channel=channel.value,
            chat_history_length=chat_history_length,
        )

    def _track_timing(
        self,
        *,
        request_id: Optional[str],
        metric_name: str,
        duration_ms: float,
    ) -> None:
        """Record a timing metric.

        Args:
            request_id: Request ID for tracking.
            metric_name: Name of the metric.
            duration_ms: Duration in milliseconds.
        """
        if self.performance_tracker and request_id:
            self.performance_tracker._record_timing(
                request_id, metric_name, duration_ms
            )

    def _record_performance_metrics(
        self,
        *,
        request_id: Optional[str],
        chunks: List[ChunkData],
        top_k: int,
        similarity_threshold: float,
        kbs_used: Set[str],
        template_str: ChatPromptTemplate,
        answer_text: str,
        context_text: str,
        sources: List[str],
    ) -> None:
        """Record comprehensive performance metrics.

        Args:
            request_id: Request ID for tracking.
            chunks: Retrieved chunks.
            top_k: Number of chunks requested.
            similarity_threshold: Similarity threshold used.
            kbs_used: Set of knowledge bases used.
            template_str: Prompt template used.
            answer_text: Generated answer.
            context_text: Combined context from chunks.
            sources: List of source file paths.
        """
        if not self.performance_tracker or not request_id:
            return

        self.performance_tracker.record_retrieval(
            request_id=request_id,
            chunks=chunks,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            knowledge_bases=list(kbs_used),
        )

        self.performance_tracker.record_generation(
            request_id=request_id,
            model_name=self.llm_config.get("model", "gpt-4o-mini"),
            temperature=self.llm_config.get("temperature", 0.0),
            prompt_text=template_str,
            response_text=answer_text,
            context_text=context_text,
        )

        # Add additional metadata
        metadata_items = [
            ("top_k", top_k),
            ("similarity_threshold", similarity_threshold),
            ("embedding_model", self.llm_config.get("embedding_model")),
            ("sources_count", len(set(sources))),
        ]
        for key, value in metadata_items:
            self.performance_tracker.add_metadata(
                request_id=request_id,
                key=key,
                value=value,
            )

    def _finalize_performance_tracking(
        self,
        *,
        request_id: Optional[str],
    ) -> None:
        """Finalize performance tracking and log summary.

        Args:
            request_id: Request ID for tracking.
        """
        if not self.performance_tracker or not request_id:
            return

        record = self.performance_tracker.end_request(request_id)
        if record:
            logger.info(
                f"Query completed - Total: {record.timing.total_pipeline_ms:.2f}ms, "
                f"Retrieval: {record.timing.retrieval_total_ms:.2f}ms, "
                f"Generation: {record.timing.llm_generation_ms:.2f}ms, "
                f"Chunks: {record.retrieval.chunks_retrieved}"
            )

    # ========================================================================
    # Main Query Method
    # ========================================================================

    def query(
        self,
        query_text: str,
        *,
        top_k: Optional[int] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        channel: str = "email",
        similarity_threshold: Optional[float] = None,
        session_id: Optional[str] = None,
        draft_mode: bool = False,
        draft_customer_query: Optional[str] = None,
        draft_current_draft: Optional[str] = None,
    ) -> QueryResult:
        """Query across all knowledge bases.

        Args:
            query_text: User query text.
            top_k: Number of chunks to retrieve (uses config default if None).
            chat_history: Previous conversation messages.
            channel: Output channel style (email or whatsapp).
            similarity_threshold: Similarity threshold for retrieving chunks.
            session_id: Optional session ID for tracking.
            draft_mode: Whether this is a draft refinement request.

        Returns:
            QueryResult with answer, sources, and metadata.

        Raises:
            Exception: If query execution fails.
        """
        # Reload configuration to get latest values
        self._load_config()

        # Apply defaults from configuration
        top_k = top_k or self.rag_config.get("top_k")
        similarity_threshold = similarity_threshold or self.rag_config.get(
            "similarity_threshold"
        )
        channel_enum = Channel.from_string(channel)

        logger.info(
            f"Processing query: '{query_text[:50]}...' "
            f"(top_k={top_k}, similarity_threshold={similarity_threshold})"
        )

        # Start performance tracking
        request_id = None
        if self.performance_tracker:
            request_id = self.performance_tracker.start_request(session_id=session_id)

        try:
            return self._execute_query(
                query_text=query_text,
                top_k=top_k,
                chat_history=chat_history,
                channel=channel_enum,
                similarity_threshold=similarity_threshold,
                session_id=session_id,
                draft_mode=draft_mode,
                draft_customer_query=draft_customer_query,
                draft_current_draft=draft_current_draft,
                request_id=request_id,
            )

        except Exception as e:
            logger.error(f"Error during query: {e}", exc_info=True)
            if self.performance_tracker and request_id:
                self.performance_tracker.record_error(
                    request_id, e, error_context="query_execution"
                )
            raise

        finally:
            self._finalize_performance_tracking(request_id=request_id)

    def _execute_query(
        self,
        *,
        query_text: str,
        top_k: int,
        chat_history: Optional[List[Dict[str, str]]],
        channel: Channel,
        similarity_threshold: float,
        session_id: Optional[str],
        draft_mode: bool,
        draft_customer_query: Optional[str],
        draft_current_draft: Optional[str],
        request_id: Optional[str],
    ) -> QueryResult:
        """Execute the query pipeline.

        Args:
            query_text: User query text.
            top_k: Number of chunks to retrieve.
            chat_history: Previous conversation messages.
            channel: Communication channel.
            similarity_threshold: Similarity threshold for retrieval.
            session_id: Session ID for tracking.
            draft_mode: Whether this is a draft refinement.
            request_id: Request ID for performance tracking.

        Returns:
            QueryResult with answer, sources, and metadata.
        """
        # Check for required clients
        if not self.embed_model or not self.llm:
            logger.warning("OpenAI API key not configured, returning error response")
            self._handle_missing_api_key_error(request_id=request_id)
            return {
                "answer": "OpenAI API key not configured.",
                "sources": [],
                "kbs_used": [],
                "chunks": [],
                "request_id": None,
            }

        # Step 1: Enhance query
        enhancement_start = time.perf_counter()
        enhancement_result = self._get_enhanced_query_and_intents(
            query_text=query_text,
            chat_history=chat_history,
            draft_mode=draft_mode,
        )
        enhanced_query = enhancement_result["enhanced_query"]
        print(enhanced_query, 'enhanced query')
        intents: List[str] = enhancement_result["intents"]

        # In draft refinement, also embed the original customer query so retrieval
        # doesn't collapse to only the latest refinement request.
        enhanced_queries: List[str] = enhanced_query
        # if draft_mode and draft_customer_query:
        #     logger.info(f"Draft customer query: {draft_customer_query}")
        #     customer_enhancement_result = self._get_enhanced_query_and_intents(
        #         query_text=draft_customer_query,
        #         chat_history=chat_history,
        #         draft_mode=False,
        #     )
        #     customer_enhanced_query = customer_enhancement_result["enhanced_query"]
        #     if customer_enhanced_query and customer_enhanced_query not in enhanced_queries:
        #         enhanced_queries.extend(customer_enhanced_query)

        #     # Merge intents to avoid over-filtering retrieval during refinement.
        #     customer_intents = customer_enhancement_result.get("intents", [])
        #     intents = list(dict.fromkeys([*intents, *customer_intents]))

        # Also embed a compact hint derived from the current draft so retrieval
        # continues to pull context for previously-covered topics.
        # if draft_mode and draft_current_draft:
            # draft_hint = self._build_draft_retrieval_hint(draft_current_draft)
            # if draft_hint and draft_hint not in enhanced_queries:
            #     enhanced_queries.append(draft_hint)
        enhancement_time_ms = (time.perf_counter() - enhancement_start) * 1000

        logger.info(f"Enhanced query: {enhanced_queries}")
        self._track_timing(
            request_id=request_id,
            metric_name="query_enhancement",
            duration_ms=enhancement_time_ms,
        )
        self._track_query_metrics(
            request_id=request_id,
            original_query=query_text,
            enhanced_query=','.join(enhanced_queries),
            channel=channel,
            chat_history_length=len(chat_history) if chat_history else 0,
        )

        # Step 2: Create retrievers
        metadata_filters = self._create_metadata_filters(intents)
        retrievers = self._create_retrievers(
            similarity_threshold=similarity_threshold,
            metadata_filters=metadata_filters,
            request_id=request_id,
        )
        hybrid_retriever = self._create_hybrid_retriever(
            retrievers=retrievers,
            top_k=top_k,
        )

        # Step 3: Build prompt template
        prompt_start = time.perf_counter()
        qa_template = self._build_qa_template(
            channel=channel,
            chat_history=chat_history,
            original_query=query_text,
            intents=intents,
            draft_mode=draft_mode,
        )
        prompt_time_ms = (time.perf_counter() - prompt_start) * 1000
        self._track_timing(
            request_id=request_id,
            metric_name="prompt_construction",
            duration_ms=prompt_time_ms,
        )

        # Step 4: Create query engine
        response_synthesizer = self._create_response_synthesizer(
            qa_template=qa_template
        )
        postprocessors = self._create_node_postprocessors(top_k=top_k)
        query_engine = RetrieverQueryEngine(
            retriever=hybrid_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors,
        )

        # Step 5: Execute query
        llm_start = time.perf_counter()
        if draft_mode:
            final_query_text  =  f"DEVELOPER REQUEST: {query_text}"
        else:
            final_query_text  =  query_text

        response = query_engine.query(
            str_or_query_bundle=QueryBundle(
                query_str=final_query_text,
                custom_embedding_strs=enhanced_queries,
            )
        )
        llm_time_ms = (time.perf_counter() - llm_start) * 1000
        self._track_timing(
            request_id=request_id,
            metric_name="llm_generation",
            duration_ms=llm_time_ms,
        )

        # Step 6: Process results
        answer_text = str(response)
        retrieval_result = self._extract_retrieval_result(response)

        # Step 7: Record metrics
        self._record_performance_metrics(
            request_id=request_id,
            chunks=retrieval_result.chunks,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            kbs_used=retrieval_result.kbs_used,
            template_str=qa_template,
            answer_text=answer_text,
            context_text=retrieval_result.context_text,
            sources=retrieval_result.sources,
        )

        return self._build_query_result(
            answer_text=answer_text,
            retrieval_result=retrieval_result,
            request_id=request_id,
        )

    def _handle_missing_api_key_error(
        self,
        *,
        request_id: Optional[str],
    ) -> None:
        """Handle the case when API key is not configured.

        Args:
            request_id: Request ID for error tracking.
        """
        if self.performance_tracker and request_id:
            self.performance_tracker.record_error(
                request_id,
                ValueError("OpenAI API key not configured"),
                error_context="initialization",
            )
            self.performance_tracker.end_request(request_id)

    # ========================================================================
    # Streaming Query
    # ========================================================================

    async def stream_query(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        chat_history: Optional[List[Dict[str, str]]] = None,
        channel: str = "email",
        similarity_threshold: Optional[float] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream query response in chunks.

        Args:
            query_text: User query text.
            top_k: Number of chunks to retrieve.
            chat_history: Previous conversation messages.
            channel: Output channel style.
            similarity_threshold: Similarity threshold for retrieval.
            session_id: Session ID for tracking.

        Yields:
            Chunks of the response text.

        Note:
            Currently wraps the synchronous query method.
            Future implementation could use true streaming from LLM.
        """
        result = await asyncio.to_thread(
            self.query,
            query_text,
            top_k=top_k,
            chat_history=chat_history,
            channel=channel,
            similarity_threshold=similarity_threshold,
            session_id=session_id,
        )

        answer = result["answer"]
        chunk_size = 50
        for i in range(0, len(answer), chunk_size):
            yield answer[i : i + chunk_size]

    # ========================================================================
    # Draft Refinement
    # ========================================================================

    def refine_draft(
        self,
        *,
        query_text: str,
        draft: DraftResponse,
        top_k: int,
        channel: str = "email",
        similarity_threshold: float,
        chat_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> QueryResult:
        """Refine a draft response based on developer feedback.

        This method prepares the draft context and refinement history,
        then executes a query in draft mode to generate an improved response.

        Args:
            query_text: The refinement request/feedback from developer.
            draft: The DraftResponse object containing current draft and history.
            top_k: Number of chunks to retrieve.
            channel: Output channel style (email or whatsapp).
            similarity_threshold: Similarity threshold for retrieval.
            chat_history: Previous conversation messages.
            session_id: Session ID for tracking.

        Returns:
            QueryResult with refined answer, sources, and metadata.
        """
        # Build refinement context messages
        self._setup_draft_refinement_context(draft=draft)

        # Execute query in draft mode
        return self.query(
            query_text,
            top_k=top_k,
            chat_history=chat_history,
            channel=channel,
            similarity_threshold=similarity_threshold,
            session_id=session_id,
            draft_mode=True,
            draft_customer_query=draft.original_query,
            draft_current_draft=draft.current_draft,
        )

    def _setup_draft_refinement_context(
        self,
        *,
        draft: DraftResponse,
    ) -> None:
        """Setup the context for draft refinement.

        Prepares the draft mode prompt template and enhancement history
        based on the draft's refinement history and current state.

        Args:
            draft: The DraftResponse object containing current draft and history.
        """
        # Start with refine draft system prompt
        messages: List[LlamaChatMessage] = [
            LlamaChatMessage(
                role="system",
                content=self.prompt_config.get("refine_draft"),
            )
        ]

        # Build enhancement history from refinement history
        enhancement_history: List[LlamaChatMessage] = []

        if draft.refinement_history:
            enhancement_history.append(
                LlamaChatMessage(
                    role="developer",
                    content="Refinement History:\n-------------------\n",
                )
            )
            for item in draft.refinement_history:
                enhancement_history.append(
                    LlamaChatMessage(
                        role=item["role"],
                        content=item["content"],
                    )
                )

        # Add current draft as assistant message
        if draft.current_draft:
            enhancement_history.append(
                LlamaChatMessage(
                    role="assistant",
                    content=(
                        f"CURRENT DRAFT RESPONSE: {draft.current_draft}\n"
                    ),
                )
            )

        # Add original query context
        if draft.original_query:
            enhancement_history.append(
                LlamaChatMessage(
                    role="user",
                    content=(
                        f"END CUSTOMER'S ORIGINAL QUERY: {draft.original_query}\n"
                    ),
                )
            )

        # Combine messages and set state
        messages.extend(enhancement_history)
        # used in query enhancement
        self._enhancement_draft_history = enhancement_history

        # used in response generation
        self._draft_mode_prompt_template = messages

    def clear_draft_context(self) -> None:
        """Clear the draft mode context.

        Should be called after draft refinement is complete to prevent
        draft context from leaking into subsequent queries.
        """
        self._draft_mode_prompt_template = None
        self._enhancement_draft_history = None
