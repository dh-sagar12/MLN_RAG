"""RAG service for querying across knowledge bases."""

import logging
import time
from typing import List, Dict, Any, Optional, AsyncIterator, Set
from llama_index.core.retrievers import QueryFusionRetriever
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    PromptTemplate,
    get_response_synthesizer,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llms import ChatMessage as LlamaChatMessage
from app.config import settings
from app.services.performance_tracker import get_performance_tracker, PerformanceTracker
from app.services.config_service import ConfigService
import asyncio
from app.services.prompt import ENHANCED_QUERY_PROMPT
from app.services.retriever import BM25Retriever, PostgresRetriever


logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG querying using LlamaIndex."""

    def __init__(self, db: Session):
        self.db = db
        self.embed_model = None
        self.llm = None
        self._load_config()

        # Initialize performance tracker
        self.performance_tracker: Optional[PerformanceTracker] = None
        if settings.performance_tracking_enabled:
            self.performance_tracker = get_performance_tracker(
                log_dir=settings.performance_log_dir,
                log_filename=settings.performance_log_filename,
                max_file_size_mb=settings.performance_max_file_size_mb,
                flush_interval=settings.performance_flush_interval,
            )
            logger.info("Performance tracking enabled")

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
                self.llm = OpenAI(
                    model=settings.openai_llm_model,
                    api_key=settings.openai_api_key,
                    temperature=settings.temperature,
                )
                logger.info(
                    f"""OpenAI clients initialized (embedding: {settings.openai_embedding_model}
                    LLM: {settings.openai_llm_model}
                    Temperature: {settings.temperature})"""
                )
                # self.reranker = SentenceTransformerRerank(
                #         model="cross-encoder/ms-marco-MiniLM-L-12-v2",
                #         top_n=10
                #     )
            except Exception as e:
                logger.error(f"Error initializing OpenAI clients: {e}", exc_info=True)
                raise
            finally:
                # Restore proxy environment variables
                for var, value in saved_proxies.items():
                    os.environ[var] = value
        else:
            logger.warning("OpenAI API key not configured")

    def _load_config(self):
        """Load dynamic configuration from database."""
        self.rag_config = ConfigService.get_rag_config(self.db)
        self.retriever_config = ConfigService.get_retriever_config(self.db)
        self.hybrid_config = ConfigService.get_hybrid_config(self.db)
        self.prompt_config = ConfigService.get_prompt_config(self.db)

    def get_enhanced_query(self, query_text, chat_history):
        """Generate enhanced query using chat history."""
        if not self.llm:
            return query_text

        # Use dynamic query enhancement prompt from config
        SYSTEM_PROMPT = self.prompt_config.get("query_enhancement")

        # Build combined history text for context resolution
        history_text = ""
        if chat_history:
            for msg in chat_history:
                history_text += f"{msg['role']}: {msg['content']}\n"

        combined_input = f"""
            Conversation history:
            {history_text}

            Latest user question:
            {query_text}
        """

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": combined_input,
            },
        ]

        chat_messages = [
            LlamaChatMessage(role=m["role"], content=m["content"]) for m in messages
        ]

        response = self.llm.chat(chat_messages)
        return str(response.message.content).strip()

    def get_system_prompt_for_channel(self, channel: str) -> str:
        """Return system prompt instructions tailored to a channel."""
        channel_key = (channel or "email").lower()

        # Use dynamic prompts from configuration
        base_prompt = self.prompt_config.get("base")

        channel_prompts = {
            "email": self.prompt_config.get("email", ""),
            "whatsapp": self.prompt_config.get("whatsapp", ""),
        }

        channel_guidance = channel_prompts.get(channel_key, channel_prompts.get("email", ""))
        if channel_guidance:
            return f"{base_prompt}\n\n{channel_guidance}"
        return base_prompt

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        channel: str = "email",
        similarity_threshold: Optional[float] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query across all knowledge bases.

        Args:
            query_text: User query
            top_k: Number of chunks to retrieve (uses config default if None)
            similarity_threshold: Similarity threshold for retrieving chunks (uses config default if None)
            channel: Output channel style (email or whatsapp)
            session_id: Optional session ID for tracking

        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Use dynamic defaults from configuration if not provided
        if top_k is None:
            top_k = self.rag_config.get("top_k") #top k for end of result
        if similarity_threshold is None:
            similarity_threshold = self.rag_config.get("similarity_threshold")
        
        logger.info(f"Processing query: '{query_text[:50]}...' (top_k={top_k}, similarity_threshold={similarity_threshold})")

        # Reload configuration to get latest values
        self._load_config()

        #PERFORMANCE TRACKING: Start performance tracking
        request_id = None
        if self.performance_tracker:
            request_id = self.performance_tracker.start_request(session_id=session_id)

        try:
            if not self.embed_model or not self.llm:
                logger.warning(
                    "OpenAI API key not configured, returning error response"
                )
                
                #PERFORMANCE TRACKING: Record error
                if self.performance_tracker and request_id:
                    self.performance_tracker.record_error(
                        request_id,
                        ValueError("OpenAI API key not configured"),
                        error_context="initialization",
                    )
                    self.performance_tracker.end_request(request_id)
                return {
                    "answer": "OpenAI API key not configured.",
                    "sources": [],
                    "kbs_used": [],
                    "chunks": [],
                }

            #PERFORMANCE TRACKING: Track query enhancement
            enhancement_start = time.perf_counter()
            enhanced_query = self.get_enhanced_query(
                query_text=query_text,
                chat_history=chat_history,
            )
            enhancement_time_ms = (time.perf_counter() - enhancement_start) * 1000

            logger.info(f"Enhanced query: {enhanced_query}")

            #PERFORMANCE TRACKING: Record query metrics
            if self.performance_tracker and request_id:
                self.performance_tracker._record_timing(
                    request_id, "query_enhancement", enhancement_time_ms
                )
                self.performance_tracker.record_query(
                    request_id=request_id,
                    original_query=query_text,
                    enhanced_query=enhanced_query,
                    channel=channel,
                    chat_history_length=len(chat_history) if chat_history else 0,
                )

            # Initialize Retriever with performance tracking and dynamic config
            vector_retriever = PostgresRetriever(
                db=self.db,
                embed_model=self.embed_model,
                similarity_threshold=similarity_threshold,
                performance_tracker=self.performance_tracker,
                request_id=request_id,
            )
            
            # Build retrievers list based on BM25 enabled setting
            retrievers = [vector_retriever]
            if self.retriever_config["bm25"]["enabled"]:
                bm25_retriever = BM25Retriever(
                    db=self.db,
                    performance_tracker=self.performance_tracker,
                    request_id=request_id,
                )
                retrievers.append(bm25_retriever)
            
            # Use dynamic hybrid retriever configuration
            hybrid_retriever = QueryFusionRetriever(
                retrievers=retrievers,
                similarity_top_k=top_k,
                num_queries=self.hybrid_config["num_queries"], # More than one if you want to generate multiple queries for similarity search,
                query_gen_prompt=self.prompt_config.get("query_enhancement"), #only used when num_queris > 1
                mode=self.hybrid_config["mode"], #options: RRF, Relative Score, Distance Base Score, Simple Score
                use_async=False,
                llm=self.llm,
            )
            #RRF is a rank aggregation method that combines rankings from multiple sources into a single, unified ranking, using the formula: score = sum over all retrievers of (1 / (k + rank)), where k is typically 60 


            # Build Prompt Template
            prompt_construction_start = time.perf_counter()
            system_message = self.get_system_prompt_for_channel(channel)
            # We need to include chat history in the prompt if provided
            history_context = ""
            if chat_history:
                for msg in chat_history:
                    history_context += f"{msg['role']}: {msg['content']}\n"

            # LlamaIndex text_qa_template expects 'context_str' and 'query_str'
            template_str = (
                f"{system_message}\n\n"
                f"Conversation History:\n{history_context}\n\n"
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, "
                "answer the query.\n"
                "Query: {query_str}\n"
                "Answer: "
            )

            qa_template = PromptTemplate(template_str)
            prompt_construction_time_ms = (
                time.perf_counter() - prompt_construction_start
            ) * 1000

            if self.performance_tracker and request_id:
                self.performance_tracker._record_timing(
                    request_id, "prompt_construction", prompt_construction_time_ms
                )

            # Configure Response Synthesizer with dynamic response mode
            response_mode = self.rag_config["response_mode"] 
            response_synthesizer = get_response_synthesizer(
                llm=self.llm,
                text_qa_template=qa_template,
                response_mode=response_mode, #OPTIONSL: refine, compact, tree_sumarize, simple_summarize, accumulate,compact_accumulate, generation, no_text, context_only.
            )

            # Configure Query Engine
            query_engine = RetrieverQueryEngine(
                retriever=hybrid_retriever,
                response_synthesizer=response_synthesizer,
            )

            #PERFORMANCE TRACKING: Execute Query with timing
            llm_generation_start = time.perf_counter()
            response = query_engine.query(enhanced_query)
            llm_generation_time_ms = (time.perf_counter() - llm_generation_start) * 1000

            if self.performance_tracker and request_id:
                self.performance_tracker._record_timing(
                    request_id, "llm_generation", llm_generation_time_ms
                )

            # Extract response and metadata
            answer_text = str(response)

            chunks = []
            kbs_used = set()
            sources = []
            context_text = ""

            for node_with_score in response.source_nodes:
                node = node_with_score.node
                score = node_with_score.score

                kb_name = node.metadata.get("kb_name", "Unknown")
                chunk_text = node.get_content()
                context_text += chunk_text + "\n\n"

                chunks.append(
                    {
                        "text": chunk_text,
                        "kb_id": node.metadata.get("kb_id"),
                        "kb_name": kb_name,
                        "similarity": score,
                        "metadata": node.metadata,
                    }
                )
                kbs_used.add(kb_name)
                if "file_path" in node.metadata:
                    sources.append(node.metadata["file_path"])
                    
                    
            #PERFORMANCE TRACKING: Record performance metrics
            self._record_performance_metrics(
                request_id=request_id,
                chunks=chunks,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                kbs_used=kbs_used,
                template_str=template_str,
                answer_text=answer_text,
                context_text=context_text,
                sources=sources,
            )

            result = {
                "answer": answer_text,
                "sources": list(set(sources)),
                "kbs_used": list(kbs_used),
                "chunks": chunks,
            }

            # Add performance tracking info to result if available
            if request_id:
                result["request_id"] = request_id

            return result

        except Exception as e:
            logger.error(f"Error during query: {e}", exc_info=True)
            if self.performance_tracker and request_id:
                self.performance_tracker.record_error(
                    request_id, e, error_context="query_execution"
                )
            raise

        finally:
            #PERFORMANCE TRACKING: End performance tracking
            if self.performance_tracker and request_id:
                record = self.performance_tracker.end_request(request_id)
                if record:
                    logger.info(
                        f"Query completed - Total: {record.timing.total_pipeline_ms:.2f}ms, "
                        f"Retrieval: {record.timing.retrieval_total_ms:.2f}ms, "
                        f"Generation: {record.timing.llm_generation_ms:.2f}ms, "
                        f"Chunks: {record.retrieval.chunks_retrieved}"
                    )

    def _record_performance_metrics(
        self,
        request_id: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        similarity_threshold: float,
        kbs_used: Set[str],
        template_str: str,
        answer_text: str,
        context_text: str,
        sources: List[str],
    ):
        """Record performance metrics."""


        # Record retrieval and generation metrics
        if self.performance_tracker and request_id:
            self.performance_tracker.record_retrieval(
                request_id=request_id,
                chunks=chunks,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                knowledge_bases=list(kbs_used),
            )

            self.performance_tracker.record_generation(
                request_id=request_id,
                model_name=settings.openai_llm_model,
                temperature=settings.temperature,
                prompt_text=template_str,
                response_text=answer_text,
                context_text=context_text,
            )

            # Add additional metadata
            self.performance_tracker.add_metadata(request_id, "top_k", top_k)
            self.performance_tracker.add_metadata(
                request_id, "similarity_threshold", similarity_threshold
            )
            self.performance_tracker.add_metadata(
                request_id, "embedding_model", settings.openai_embedding_model
            )
            self.performance_tracker.add_metadata(
                request_id, "sources_count", len(set(sources))
            )




    async def stream_query(self, query_text: str, top_k: int = 5) -> AsyncIterator[str]:
        """Stream query response."""
        # For streaming, we need to setup the engine similar to query()
        # This is a bit inefficient to re-init every time, but necessary for dynamic prompts (channel/history)
        # In a real app, we might cache engines or use a factory.

        # NOTE: For simplicity in this refactor, we will just call the sync query
        # because the method signature doesn't pass channel/history which we need for the prompt!
        # If we want true streaming with the correct prompt, we'd need those args here too.
        # Assuming defaults for now or wrapping sync as before.

        # To do it properly:
        # 1. We need channel/history in stream_query signature (breaking change?)
        # 2. Or we assume default channel.

        # Let's wrap sync for now to avoid breaking signature,
        # OR use a default engine.

        result = await asyncio.to_thread(self.query, query_text, top_k)
        answer = result["answer"]
        chunk_size = 50
        for i in range(0, len(answer), chunk_size):
            yield answer[i : i + chunk_size]
