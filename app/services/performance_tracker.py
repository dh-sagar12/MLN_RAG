"""
Performance Tracking Service for RAG Pipeline.

This service provides comprehensive performance monitoring and metrics collection
for the Retrieval-Augmented Generation (RAG) system. It tracks:

- Retrieval Performance: timing, chunks, similarity scores, knowledge bases
- Generation Performance: LLM metrics, token usage, response times
- Overall Pipeline: end-to-end latency, throughput
- Quality Metrics: context utilization, relevance indicators

Industry-standard metrics implemented:
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG)
- Precision@K and Recall@K
- Token efficiency and cost estimation
- Latency percentiles
"""

import os
import json
import time
import uuid
import logging
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from collections import defaultdict
import statistics
import atexit

logger = logging.getLogger(__name__)


@dataclass
class TimingMetrics:
    """Timing metrics for various pipeline stages."""
    query_enhancement_ms: float = 0.0
    embedding_generation_ms: float = 0.0
    vector_search_ms: float = 0.0
    retrieval_total_ms: float = 0.0
    prompt_construction_ms: float = 0.0
    llm_generation_ms: float = 0.0
    response_synthesis_ms: float = 0.0
    total_pipeline_ms: float = 0.0


@dataclass
class RetrievalMetrics:
    """Metrics related to document retrieval."""
    chunks_retrieved: int = 0
    chunks_above_threshold: int = 0
    top_k_requested: int = 0
    similarity_threshold: float = 0.0
    similarity_scores: List[float] = field(default_factory=list)
    avg_similarity_score: float = 0.0
    min_similarity_score: float = 0.0
    max_similarity_score: float = 0.0
    knowledge_bases_used: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    chunk_sources: List[Dict[str, Any]] = field(default_factory=list)
    # Quality indicators
    reciprocal_rank: float = 0.0  # MRR component
    ndcg_score: float = 0.0  # Normalized DCG
    precision_at_k: float = 0.0
    context_coverage: float = 0.0  # % of query terms found in context


@dataclass
class GenerationMetrics:
    """Metrics related to LLM generation."""
    model_name: str = ""
    temperature: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    prompt_length_chars: int = 0
    response_length_chars: int = 0
    context_length_chars: int = 0
    # Efficiency metrics
    tokens_per_second: float = 0.0
    cost_estimate_usd: float = 0.0
    # Quality indicators
    response_contains_context: bool = False
    context_utilization_ratio: float = 0.0


@dataclass
class QueryMetrics:
    """Metrics related to the query itself."""
    original_query: str = ""
    enhanced_query: str = ""
    query_length_chars: int = 0
    query_word_count: int = 0
    query_was_enhanced: bool = False
    enhancement_similarity: float = 0.0  # How different enhanced is from original
    channel: str = ""
    chat_history_length: int = 0


@dataclass 
class SessionMetrics:
    """Session-level tracking metrics."""
    session_id: str = ""
    request_id: str = ""
    timestamp: str = ""
    timestamp_utc: str = ""
    user_agent: Optional[str] = None
    client_ip: Optional[str] = None


@dataclass
class PerformanceRecord:
    """Complete performance record for a single RAG query."""
    # Identifiers
    session: SessionMetrics = field(default_factory=SessionMetrics)
    # Metrics
    query: QueryMetrics = field(default_factory=QueryMetrics)
    timing: TimingMetrics = field(default_factory=TimingMetrics)
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    # Status
    success: bool = True
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """
    Comprehensive performance tracking for RAG pipeline.
    
    Features:
    - Thread-safe metric collection
    - JSON file logging with rotation
    - Real-time statistics aggregation
    - Configurable metric granularity
    """

    # OpenAI pricing (approximate, update as needed)
    TOKEN_COSTS = {
        "gpt-4o": {"input": 0.0025 / 1000, "output": 0.01 / 1000},
        "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
        "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
        "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
        "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
        "text-embedding-3-small": {"input": 0.00002 / 1000, "output": 0},
        "text-embedding-3-large": {"input": 0.00013 / 1000, "output": 0},
        "text-embedding-ada-002": {"input": 0.0001 / 1000, "output": 0},
    }

    def __init__(
        self,
        log_dir: str = "logs/performance",
        log_filename: str = "performance.json",
        aggregate_filename: str = "aggregate_stats.json",
        max_file_size_mb: int = 50,
        enable_aggregation: bool = True,
        flush_interval: int = 1,  # Flush immediately for real-time logging
    ):
        """
        Initialize the performance tracker.
        
        Args:
            log_dir: Directory for performance logs
            log_filename: Name of the performance log file
            aggregate_filename: Name of the aggregate statistics file
            max_file_size_mb: Max size before rotating log file
            enable_aggregation: Whether to compute aggregate statistics
            flush_interval: Number of records before flushing to disk
        """
        self.log_dir = Path(log_dir)
        self.log_filename = log_filename
        self.aggregate_filename = aggregate_filename
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.enable_aggregation = enable_aggregation
        self.flush_interval = flush_interval
        
        # Thread-safe components
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._buffer: List[Dict[str, Any]] = []
        self._record_count = 0
        
        # Active tracking contexts
        self._active_records: Dict[str, PerformanceRecord] = {}
        self._timers: Dict[str, Dict[str, float]] = {}
        
        # Aggregate statistics (in-memory)
        self._aggregate_stats = defaultdict(list)
        
        # Ensure log directory exists
        self._ensure_log_dir()
        
        # Register cleanup on exit to flush any remaining buffered records
        atexit.register(self._cleanup)
        
        logger.info(f"PerformanceTracker initialized. Log dir: {self.log_dir}")

    def _cleanup(self) -> None:
        """Cleanup handler called on program exit to flush remaining records."""
        try:
            self.flush()
            logger.info("Performance tracker flushed on exit")
        except Exception as e:
            logger.error(f"Error during performance tracker cleanup: {e}")

    def _ensure_log_dir(self) -> None:
        """Create log directory if it doesn't exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured log directory exists: {self.log_dir}")

    def _get_log_path(self) -> Path:
        """Get the current log file path."""
        return self.log_dir / self.log_filename

    def _get_aggregate_path(self) -> Path:
        """Get the aggregate statistics file path."""
        return self.log_dir / self.aggregate_filename

    def _rotate_log_if_needed(self) -> None:
        """Rotate log file if it exceeds max size."""
        log_path = self._get_log_path()
        if log_path.exists() and log_path.stat().st_size > self.max_file_size_bytes:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_name = f"performance_{timestamp}.json"
            rotated_path = self.log_dir / rotated_name
            log_path.rename(rotated_path)
            logger.info(f"Rotated performance log to: {rotated_name}")

    # =========================================================================
    # Context Management - Start/Stop Tracking
    # =========================================================================

    def start_request(
        self,
        session_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> str:
        """
        Start tracking a new request.
        
        Returns:
            request_id: Unique identifier for this request
        """
        request_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        record = PerformanceRecord(
            session=SessionMetrics(
                session_id=session_id or str(uuid.uuid4()),
                request_id=request_id,
                timestamp=now.isoformat(),
                timestamp_utc=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                user_agent=user_agent,
                client_ip=client_ip,
            )
        )
        
        with self._lock:
            self._active_records[request_id] = record
            self._timers[request_id] = {"pipeline_start": time.perf_counter()}
        
        logger.debug(f"Started tracking request: {request_id}")
        return request_id

    def end_request(self, request_id: str) -> Optional[PerformanceRecord]:
        """
        End tracking for a request and persist the record.
        
        Returns:
            The completed PerformanceRecord
        """
        with self._lock:
            if request_id not in self._active_records:
                logger.warning(f"Request {request_id} not found in active records")
                return None
            
            record = self._active_records.pop(request_id)
            timers = self._timers.pop(request_id, {})
        
        # Calculate total pipeline time
        if "pipeline_start" in timers:
            record.timing.total_pipeline_ms = (
                time.perf_counter() - timers["pipeline_start"]
            ) * 1000
        
        # Compute derived metrics
        self._compute_derived_metrics(record)
        
        # Persist record
        self._persist_record(record)
        
        # Update aggregates
        if self.enable_aggregation:
            self._update_aggregates(record)
        
        logger.debug(f"Ended tracking request: {request_id}")
        return record

    @contextmanager
    def track_request(
        self,
        session_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        client_ip: Optional[str] = None,
    ):
        """
        Context manager for tracking a complete request.
        
        Usage:
            with tracker.track_request() as request_id:
                # Your RAG pipeline code
                tracker.record_query(request_id, ...)
        """
        request_id = self.start_request(session_id, user_agent, client_ip)
        try:
            yield request_id
        except Exception as e:
            self.record_error(request_id, e)
            raise
        finally:
            self.end_request(request_id)

    # =========================================================================
    # Timer Utilities
    # =========================================================================

    @contextmanager
    def time_operation(self, request_id: str, operation_name: str):
        """
        Context manager for timing an operation.
        
        Usage:
            with tracker.time_operation(request_id, "embedding_generation"):
                # Embedding generation code
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_timing(request_id, operation_name, elapsed_ms)

    def start_timer(self, request_id: str, operation_name: str) -> None:
        """Start a named timer for manual timing control."""
        with self._lock:
            if request_id in self._timers:
                self._timers[request_id][f"{operation_name}_start"] = time.perf_counter()

    def stop_timer(self, request_id: str, operation_name: str) -> float:
        """Stop a named timer and record the elapsed time."""
        with self._lock:
            if request_id not in self._timers:
                return 0.0
            
            start_key = f"{operation_name}_start"
            if start_key not in self._timers[request_id]:
                return 0.0
            
            elapsed_ms = (
                time.perf_counter() - self._timers[request_id].pop(start_key)
            ) * 1000
        
        self._record_timing(request_id, operation_name, elapsed_ms)
        return elapsed_ms

    def _record_timing(self, request_id: str, operation_name: str, elapsed_ms: float) -> None:
        """Record timing for a specific operation."""
        with self._lock:
            if request_id not in self._active_records:
                return
            
            record = self._active_records[request_id]
            timing_map = {
                "query_enhancement": "query_enhancement_ms",
                "embedding_generation": "embedding_generation_ms",
                "vector_search": "vector_search_ms",
                "retrieval_total": "retrieval_total_ms",
                "prompt_construction": "prompt_construction_ms",
                "llm_generation": "llm_generation_ms",
                "response_synthesis": "response_synthesis_ms",
            }
            
            if operation_name in timing_map:
                setattr(record.timing, timing_map[operation_name], elapsed_ms)
            else:
                # Store in metadata for custom timings
                record.metadata[f"timing_{operation_name}_ms"] = elapsed_ms

    # =========================================================================
    # Record Metrics
    # =========================================================================

    def record_query(
        self,
        request_id: str,
        original_query: str,
        enhanced_query: Optional[str] = None,
        channel: str = "email",
        chat_history_length: int = 0,
    ) -> None:
        """Record query-related metrics."""
        with self._lock:
            if request_id not in self._active_records:
                return
            
            record = self._active_records[request_id]
            record.query.original_query = original_query
            record.query.enhanced_query = enhanced_query or original_query
            record.query.query_length_chars = len(original_query)
            record.query.query_word_count = len(original_query.split())
            record.query.query_was_enhanced = enhanced_query is not None and enhanced_query != original_query
            record.query.channel = channel
            record.query.chat_history_length = chat_history_length
            
            # Calculate enhancement similarity (simple Jaccard)
            if record.query.query_was_enhanced:
                original_words = set(original_query.lower().split())
                enhanced_words = set(record.query.enhanced_query.lower().split())
                intersection = original_words & enhanced_words
                union = original_words | enhanced_words
                record.query.enhancement_similarity = len(intersection) / len(union) if union else 1.0

    def record_retrieval(
        self,
        request_id: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        similarity_threshold: float,
        knowledge_bases: Optional[List[str]] = None,
    ) -> None:
        """Record retrieval-related metrics."""
        with self._lock:
            if request_id not in self._active_records:
                return
            
            record = self._active_records[request_id]
            
            # Basic counts
            record.retrieval.chunks_retrieved = len(chunks)
            record.retrieval.top_k_requested = top_k
            record.retrieval.similarity_threshold = similarity_threshold
            
            # Extract similarity scores and chunk info
            scores = []
            chunk_ids = []
            chunk_sources = []
            kb_set = set()
            
            for chunk in chunks:
                score = chunk.get("similarity", chunk.get("score", 0))
                scores.append(float(score))
                
                # Extract chunk ID
                chunk_id = chunk.get("metadata", {}).get("node_id", "")
                if not chunk_id:
                    # Generate a hash-based ID if none exists
                    chunk_text = chunk.get("text", "")[:100]
                    chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
                chunk_ids.append(chunk_id)
                
                # Extract source info
                kb_name = chunk.get("kb_name", chunk.get("metadata", {}).get("kb_name", "unknown"))
                kb_set.add(kb_name)
                
                chunk_sources.append({
                    "chunk_id": chunk_id,
                    "kb_name": kb_name,
                    "kb_id": chunk.get("kb_id", chunk.get("metadata", {}).get("kb_id", "")),
                    "similarity": score,
                    "file_path": chunk.get("metadata", {}).get("file_path", ""),
                    "text_preview": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
                })
            
            record.retrieval.similarity_scores = scores
            record.retrieval.chunk_ids = chunk_ids
            record.retrieval.chunk_sources = chunk_sources
            record.retrieval.knowledge_bases_used = knowledge_bases or list(kb_set)
            
            # Calculate score statistics
            if scores:
                record.retrieval.avg_similarity_score = statistics.mean(scores)
                record.retrieval.min_similarity_score = min(scores)
                record.retrieval.max_similarity_score = max(scores)
                record.retrieval.chunks_above_threshold = sum(1 for s in scores if s >= similarity_threshold)
                
                # Calculate MRR (assuming first relevant doc is best)
                if scores[0] >= similarity_threshold:
                    record.retrieval.reciprocal_rank = 1.0
                else:
                    for i, score in enumerate(scores):
                        if score >= similarity_threshold:
                            record.retrieval.reciprocal_rank = 1.0 / (i + 1)
                            break
                
                # Calculate nDCG
                record.retrieval.ndcg_score = self._calculate_ndcg(scores, k=top_k)
                
                # Precision@K
                record.retrieval.precision_at_k = record.retrieval.chunks_above_threshold / len(scores) if scores else 0

    def record_generation(
        self,
        request_id: str,
        model_name: str,
        temperature: float,
        prompt_text: str,
        response_text: str,
        context_text: str = "",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        """Record generation-related metrics."""
        with self._lock:
            if request_id not in self._active_records:
                return
            
            record = self._active_records[request_id]
            
            record.generation.model_name = model_name
            record.generation.temperature = temperature
            record.generation.prompt_length_chars = len(prompt_text)
            record.generation.response_length_chars = len(response_text)
            record.generation.context_length_chars = len(context_text)
            
            # Token counting (estimate if not provided)
            if input_tokens is not None:
                record.generation.input_tokens = input_tokens
            else:
                # Rough estimate: ~4 chars per token for English
                record.generation.input_tokens = len(prompt_text) // 4
            
            if output_tokens is not None:
                record.generation.output_tokens = output_tokens
            else:
                record.generation.output_tokens = len(response_text) // 4
            
            record.generation.total_tokens = (
                record.generation.input_tokens + record.generation.output_tokens
            )
            
            # Calculate tokens per second
            if record.timing.llm_generation_ms > 0:
                record.generation.tokens_per_second = (
                    record.generation.output_tokens / (record.timing.llm_generation_ms / 1000)
                )
            
            # Estimate cost
            record.generation.cost_estimate_usd = self._estimate_cost(
                model_name,
                record.generation.input_tokens,
                record.generation.output_tokens,
            )
            
            # Context utilization
            if context_text:
                # Check if key phrases from context appear in response
                context_phrases = set(context_text.lower().split())
                response_phrases = set(response_text.lower().split())
                overlap = context_phrases & response_phrases
                record.generation.context_utilization_ratio = (
                    len(overlap) / len(context_phrases) if context_phrases else 0
                )
                record.generation.response_contains_context = len(overlap) > 0

    def record_error(
        self,
        request_id: str,
        error: Exception,
        error_context: Optional[str] = None,
    ) -> None:
        """Record an error that occurred during processing."""
        with self._lock:
            if request_id not in self._active_records:
                return
            
            record = self._active_records[request_id]
            record.success = False
            record.error_type = type(error).__name__
            record.error_message = str(error)
            if error_context:
                record.metadata["error_context"] = error_context

    def add_metadata(
        self,
        request_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Add custom metadata to a request record."""
        with self._lock:
            if request_id not in self._active_records:
                return
            self._active_records[request_id].metadata[key] = value

    # =========================================================================
    # Derived Metrics Computation
    # =========================================================================

    def _calculate_ndcg(self, scores: List[float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not scores:
            return 0.0
        
        # DCG
        dcg = sum(
            score / (i + 2) * 1.4426950408889634  # log2(i+2)
            for i, score in enumerate(scores[:k])
        )
        
        # Ideal DCG (sorted scores)
        ideal_scores = sorted(scores, reverse=True)
        idcg = sum(
            score / (i + 2) * 1.4426950408889634
            for i, score in enumerate(ideal_scores[:k])
        )
        
        return dcg / idcg if idcg > 0 else 0.0

    def _estimate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost in USD based on token usage."""
        # Normalize model name
        model_key = model_name.lower()
        for key in self.TOKEN_COSTS:
            if key in model_key:
                costs = self.TOKEN_COSTS[key]
                return (
                    input_tokens * costs["input"] +
                    output_tokens * costs["output"]
                )
        
        # Default fallback (gpt-4o-mini pricing)
        return input_tokens * 0.00015 / 1000 + output_tokens * 0.0006 / 1000

    def _compute_derived_metrics(self, record: PerformanceRecord) -> None:
        """Compute derived metrics that depend on other metrics."""
        # Context coverage (query terms found in retrieved chunks)
        if record.query.original_query and record.retrieval.chunk_sources:
            query_words = set(record.query.original_query.lower().split())
            context_text = " ".join(
                c.get("text_preview", "") for c in record.retrieval.chunk_sources
            ).lower()
            context_words = set(context_text.split())
            
            overlap = query_words & context_words
            record.retrieval.context_coverage = (
                len(overlap) / len(query_words) if query_words else 0
            )
        
        # Ensure retrieval total is computed
        if record.timing.retrieval_total_ms == 0:
            record.timing.retrieval_total_ms = (
                record.timing.embedding_generation_ms +
                record.timing.vector_search_ms
            )

    # =========================================================================
    # Persistence
    # =========================================================================

    def _persist_record(self, record: PerformanceRecord) -> None:
        """Persist a performance record to the log file."""
        record_dict = self._record_to_dict(record)
        
        with self._write_lock:
            self._buffer.append(record_dict)
            self._record_count += 1
            
            if len(self._buffer) >= self.flush_interval:
                self._flush_buffer()

    def _record_to_dict(self, record: PerformanceRecord) -> Dict[str, Any]:
        """Convert a PerformanceRecord to a dictionary."""
        return {
            "session": asdict(record.session),
            "query": asdict(record.query),
            "timing": asdict(record.timing),
            "retrieval": asdict(record.retrieval),
            "generation": asdict(record.generation),
            "success": record.success,
            "error_message": record.error_message,
            "error_type": record.error_type,
            "metadata": record.metadata,
        }

    def _flush_buffer(self) -> None:
        """Flush the buffer to disk."""
        if not self._buffer:
            return
        
        self._rotate_log_if_needed()
        log_path = self._get_log_path()
        
        try:
            # Read existing records or initialize empty list
            existing_records = []
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            existing_records = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    # File is corrupted or empty, start fresh
                    logger.warning(f"Could not parse existing log file, starting fresh")
                    existing_records = []
            
            # Append new records
            existing_records.extend(self._buffer)
            
            # Write back
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(existing_records, f, indent=2, default=str)
            
            logger.debug(f"Flushed {len(self._buffer)} records to {log_path}")
            self._buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing performance records: {e}", exc_info=True)

    def flush(self) -> None:
        """Manually flush the buffer to disk."""
        with self._write_lock:
            self._flush_buffer()
            if self.enable_aggregation:
                self._save_aggregates()

    # =========================================================================
    # Aggregate Statistics
    # =========================================================================

    def _update_aggregates(self, record: PerformanceRecord) -> None:
        """Update aggregate statistics with a new record."""
        # Timing aggregates
        self._aggregate_stats["total_pipeline_ms"].append(record.timing.total_pipeline_ms)
        self._aggregate_stats["retrieval_total_ms"].append(record.timing.retrieval_total_ms)
        self._aggregate_stats["llm_generation_ms"].append(record.timing.llm_generation_ms)
        
        # Retrieval aggregates
        self._aggregate_stats["chunks_retrieved"].append(record.retrieval.chunks_retrieved)
        self._aggregate_stats["avg_similarity"].append(record.retrieval.avg_similarity_score)
        
        # Generation aggregates
        self._aggregate_stats["total_tokens"].append(record.generation.total_tokens)
        self._aggregate_stats["cost_usd"].append(record.generation.cost_estimate_usd)
        
        # Success rate
        self._aggregate_stats["success"].append(1 if record.success else 0)
        
        # Limit memory usage (keep last 10000 records for aggregates)
        max_aggregate_samples = 10000
        for key in self._aggregate_stats:
            if len(self._aggregate_stats[key]) > max_aggregate_samples:
                self._aggregate_stats[key] = self._aggregate_stats[key][-max_aggregate_samples:]

    def _save_aggregates(self) -> None:
        """Save aggregate statistics to file."""
        if not self._aggregate_stats:
            return
        
        try:
            stats = {}
            for key, values in self._aggregate_stats.items():
                if not values:
                    continue
                
                stats[key] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }
                
                if len(values) > 1:
                    stats[key]["stdev"] = statistics.stdev(values)
                    # Percentiles
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    stats[key]["p50"] = sorted_values[int(n * 0.5)]
                    stats[key]["p90"] = sorted_values[int(n * 0.9)]
                    stats[key]["p95"] = sorted_values[int(n * 0.95)]
                    stats[key]["p99"] = sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1]
            
            stats["last_updated"] = datetime.now(timezone.utc).isoformat()
            stats["total_requests"] = self._record_count
            
            with open(self._get_aggregate_path(), "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            
            logger.debug(f"Saved aggregate statistics to {self._get_aggregate_path()}")
            
        except Exception as e:
            logger.error(f"Error saving aggregate statistics: {e}", exc_info=True)

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get current aggregate statistics."""
        stats = {}
        for key, values in self._aggregate_stats.items():
            if not values:
                continue
            stats[key] = {
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values) if values else 0,
            }
        return stats

    # =========================================================================
    # Query/Analysis Methods
    # =========================================================================

    def get_recent_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the most recent performance records."""
        log_path = self._get_log_path()
        if not log_path.exists():
            return []
        
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            return records[-limit:]
        except Exception as e:
            logger.error(f"Error reading recent records: {e}")
            return []

    def get_slow_queries(
        self,
        threshold_ms: float = 5000,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get queries that exceeded the latency threshold."""
        records = self.get_recent_records(limit=1000)
        slow = [
            r for r in records
            if r.get("timing", {}).get("total_pipeline_ms", 0) > threshold_ms
        ]
        return sorted(
            slow,
            key=lambda x: x.get("timing", {}).get("total_pipeline_ms", 0),
            reverse=True,
        )[:limit]

    def get_failed_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get queries that failed."""
        records = self.get_recent_records(limit=1000)
        failed = [r for r in records if not r.get("success", True)]
        return failed[-limit:]


# =============================================================================
# Global Singleton Instance
# =============================================================================

_tracker_instance: Optional[PerformanceTracker] = None


def get_performance_tracker(
    log_dir: str = "logs/performance",
    **kwargs,
) -> PerformanceTracker:
    """
    Get the global performance tracker instance.
    
    Creates a new instance if one doesn't exist.
    """
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PerformanceTracker(log_dir=log_dir, **kwargs)
    return _tracker_instance


def reset_performance_tracker() -> None:
    """Reset the global performance tracker instance."""
    global _tracker_instance
    if _tracker_instance:
        _tracker_instance.flush()
    _tracker_instance = None

