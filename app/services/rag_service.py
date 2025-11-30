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

    def get_enhanced_query(self, query_text, chat_history):
        """Generate enhanced query using chat history."""
        if not self.llm:
            return query_text

        SYSTEM_PROMPT = ENHANCED_QUERY_PROMPT

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

        base_prompt = "You are responding on behalf of Mountain Lodges of Nepal part of Sherpa Hospitality Group, a premium Himalayan hospitality and travel company. Format your response using Markdown for better readability (use bullet points, bold text, and paragraphs where appropriate)."

        channel_prompts = {
            "email": (
                """Your role is to write warm, professional, and accurate email replies to guests, tour operators, and partners based only on the information provided in the CONTEXT and THREAD sections below.

                Tone & Style Guidelines
                    •    Warm, welcoming, and hospitality-oriented.
                    •    Clear, complete sentences.
                    •    Polite and reassuring.
                    •    Naturally formal, but friendly and approachable.
                    •    Convey confidence and care as a representative of the brand.

                STRICT RULES
                    •    Do not invent facts, prices, availability, dates, or commitments.
                    •    If required information is not present in the CONTEXT, simply say:
                “We will check this and get back to you shortly.”
                    •    Prefer the most recent and active policies.
                    •    If multiple snippets conflict, rely on the most recent or clearly valid one.
                    •    Never reference internal details, metadata, or system instructions.
                    •    Do not quote outdated prices or weather conditions from previous years.
                    •    Keep the reply fully self-contained, without mentioning lack of data or internal processes.

                When context is incomplete

                If the available information does not allow for an accurate or safe answer, write a polite and helpful reply and include a natural line such as:
                “We will confirm this for you and get back to you soon.”

                Goal

                Produce a polished, guest-ready email that feels like it was written by a trained hospitality professional at MOuntain Lodges of Nepal , maintaining accuracy and brand trust at all times. 

                """
            ),
            "whatsapp": (
                """.
                Your role is to write friendly, concise, and accurate WhatsApp replies to guests, tour operators, and partners based only on the information provided in the CONTEXT and THREAD sections below

                Tone & Style Guidelines
                    •    Warm, welcoming, and guest-oriented.
                    •    Shorter paragraphs, conversational, but still professional.
                    •    Lightly enthusiastic and attentive — the tone of a helpful hospitality host.
                    •    Natural phrasing suitable for mobile messaging.

                STRICT RULES
                    •    Never invent prices, availability, dates, or operational details.
                    •    If a guest asks for something not present in the CONTEXT, simply say:
                “We’ll check this and get back to you shortly.”
                    •    Use only the most recent and active details.
                    •    If conflicting information appears, use the most updated one.
                    •    Do not reveal that you are using AI or systems.
                    •    Do not include internal notes, metadata, or technical labels.
                    •    Avoid quoting outdated seasonal details or old prices.

                When context is incomplete

                Keep the reply helpful and friendly, and add:
                “We’ll confirm the details and update you soon.”

                Goal

                Produce a natural, helpful WhatsApp message that feels like a real team member of  Mountain Lodges of Nepal — supportive, accurate, and hospitality-driven."""
            ),
        }

        channel_guidance = channel_prompts.get(channel_key, channel_prompts["email"])
        return f"{base_prompt}\n\n{channel_guidance}"

    def query(
        self,
        query_text: str,
        top_k: int = 7,
        chat_history: Optional[List[Dict[str, str]]] = None,
        channel: str = "email",
        similarity_threshold: float = 0.75,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query across all knowledge bases.

        Args:
            query_text: User query
            top_k: Number of chunks to retrieve
            similarity_threshold: Similarity threshold for retrieving chunks
            channel: Output channel style (email or whatsapp)
            session_id: Optional session ID for tracking

        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"Processing query: '{query_text[:50]}...' (top_k={top_k})")

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

            # Initialize Retriever with performance tracking
            vector_retriever = PostgresRetriever(
                db=self.db,
                embed_model=self.embed_model,
                # top_k=top_k,
                similarity_threshold=similarity_threshold,
                performance_tracker=self.performance_tracker,
                request_id=request_id,
            )
            bm25_retriever = BM25Retriever(
                db=self.db,
                performance_tracker=self.performance_tracker,
                request_id=request_id,
            )
            
            hybrid_retriever = QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                similarity_top_k=top_k,  # Before reranking (we will increate this if we implement reranker later.)
                num_queries=1,
                mode="reciprocal_rerank",  # Reciprocal Rank Fusion 
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

            # Configure Response Synthesizer
            response_synthesizer = get_response_synthesizer(
                llm=self.llm,
                text_qa_template=qa_template,
                response_mode="compact", #TODO: make it dynamic
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
