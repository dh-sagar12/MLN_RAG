"""Performance tracking routes."""

import json
import logging
from pathlib import Path
from collections import defaultdict
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.templating import Jinja2Templates
from sqlalchemy import select

from app.config import settings
from app.services.performance_tracker import get_performance_tracker
from app.database import SessionLocal
from app.models.embedding import Embedding
from app.models.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


async def performance_page(request: Request):
    """Render the performance tracking dashboard."""
    return templates.TemplateResponse(
        "performance.html",
        {"request": request}
    )


async def get_performance_records(request: Request):
    """Get performance records as JSON."""
    try:
        limit = int(request.query_params.get("limit", 100))
        tracker = get_performance_tracker(log_dir=settings.performance_log_dir)
        records = tracker.get_recent_records(limit=limit)
        
        # Sort by timestamp descending (most recent first)
        records.sort(
            key=lambda x: x.get("session", {}).get("timestamp", ""),
            reverse=True
        )
        
        return JSONResponse({"records": records, "total": len(records)})
    except Exception as e:
        logger.error(f"Error fetching performance records: {e}", exc_info=True)
        return JSONResponse({"error": str(e), "records": []}, status_code=500)


async def get_aggregate_stats(request: Request):
    """Get aggregate statistics."""
    try:
        tracker = get_performance_tracker(log_dir=settings.performance_log_dir)
        stats = tracker.get_aggregate_stats()
        return JSONResponse(stats)
    except Exception as e:
        logger.error(f"Error fetching aggregate stats: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_slow_queries(request: Request):
    """Get slow queries above threshold."""
    try:
        threshold_ms = float(request.query_params.get("threshold", 5000))
        limit = int(request.query_params.get("limit", 50))
        
        tracker = get_performance_tracker(log_dir=settings.performance_log_dir)
        slow = tracker.get_slow_queries(threshold_ms=threshold_ms, limit=limit)
        
        return JSONResponse({"slow_queries": slow, "threshold_ms": threshold_ms})
    except Exception as e:
        logger.error(f"Error fetching slow queries: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_failed_queries(request: Request):
    """Get failed queries."""
    try:
        limit = int(request.query_params.get("limit", 50))
        tracker = get_performance_tracker(log_dir=settings.performance_log_dir)
        failed = tracker.get_failed_queries(limit=limit)
        
        return JSONResponse({"failed_queries": failed, "total": len(failed)})
    except Exception as e:
        logger.error(f"Error fetching failed queries: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# Metric explanations for the frontend
METRIC_EXPLANATIONS = {
    # Timing Metrics
    "query_enhancement_ms": {
        "name": "Query Enhancement Time",
        "description": "Time taken by the LLM to rewrite/enhance the user's query for better semantic search. This uses the chat history to resolve context and references.",
        "unit": "milliseconds",
        "good_range": "< 2000ms"
    },
    "embedding_generation_ms": {
        "name": "Embedding Generation Time", 
        "description": "Time to convert the query text into a vector embedding using OpenAI's embedding model. This vector is used to search the knowledge base.",
        "unit": "milliseconds",
        "good_range": "< 500ms"
    },
    "vector_search_ms": {
        "name": "Vector Search Time",
        "description": "Time to search the PostgreSQL database with pgvector to find the most similar document chunks. Uses HNSW index for fast approximate nearest neighbor search.",
        "unit": "milliseconds", 
        "good_range": "< 100ms"
    },
    "bm25_search_ms": {
        "name": "BM25 Search Time",
        "description": "Time to perform full-text search using PostgreSQL's ts_rank_cd (BM25-like ranking). This is keyword-based sparse retrieval as opposed to semantic vector search.",
        "unit": "milliseconds",
        "good_range": "< 100ms"
    },
    "retrieval_total_ms": {
        "name": "Total Retrieval Time",
        "description": "Combined time for embedding generation + vector search. This is the complete time to fetch relevant context from the knowledge base.",
        "unit": "milliseconds",
        "good_range": "< 600ms"
    },
    "llm_generation_ms": {
        "name": "LLM Generation Time",
        "description": "Time for the LLM (GPT model) to generate the final response based on the retrieved context and query.",
        "unit": "milliseconds",
        "good_range": "< 5000ms"
    },
    "total_pipeline_ms": {
        "name": "Total Pipeline Time",
        "description": "End-to-end time from receiving the query to returning the response. Includes all stages: enhancement, retrieval, and generation.",
        "unit": "milliseconds",
        "good_range": "< 8000ms"
    },
    
    # Retrieval Quality Metrics
    "reciprocal_rank": {
        "name": "Reciprocal Rank (RR)",
        "description": "Measures how quickly the first relevant result appears. RR = 1/rank of first relevant result. A score of 1.0 means the first result was relevant; 0.5 means the second result was first relevant.",
        "unit": "score (0-1)",
        "good_range": "> 0.8"
    },
    "ndcg_score": {
        "name": "Normalized DCG (nDCG)",
        "description": "Measures ranking quality considering position and relevance of ALL retrieved documents. Higher scores mean more relevant documents appear earlier. Industry-standard IR metric.",
        "unit": "score (0-1)",
        "good_range": "> 0.7"
    },
    "precision_at_k": {
        "name": "Precision@K",
        "description": "Fraction of retrieved documents that are relevant (above similarity threshold). precision@K = relevant_retrieved / total_retrieved. Higher is better.",
        "unit": "ratio (0-1)",
        "good_range": "> 0.7"
    },
    "context_coverage": {
        "name": "Context Coverage",
        "description": "Percentage of query words that appear in the retrieved context. Indicates how well the retrieval matches the query topic. Low coverage may indicate retrieval misses.",
        "unit": "ratio (0-1)",
        "good_range": "> 0.3"
    },
    "avg_similarity_score": {
        "name": "Average Similarity",
        "description": "Mean cosine similarity score across all retrieved chunks. Higher scores indicate stronger semantic match between query and retrieved content.",
        "unit": "score (0-1)",
        "good_range": "> 0.75"
    },
    
    # Generation Metrics
    "tokens_per_second": {
        "name": "Tokens Per Second",
        "description": "LLM output generation speed. Higher values indicate faster response generation. Varies by model and load.",
        "unit": "tokens/sec",
        "good_range": "> 20"
    },
    "context_utilization_ratio": {
        "name": "Context Utilization",
        "description": "Percentage of context words that appear in the response. Indicates how much of the retrieved information the LLM actually used in its answer.",
        "unit": "ratio (0-1)",
        "good_range": "> 0.05"
    },
    "cost_estimate_usd": {
        "name": "Estimated Cost",
        "description": "Approximate cost in USD based on token usage and OpenAI pricing. Helps track API expenses per query.",
        "unit": "USD",
        "good_range": "< $0.01"
    },
    
    # Query Metrics
    "enhancement_similarity": {
        "name": "Enhancement Similarity",
        "description": "Jaccard similarity between original and enhanced query (word overlap). Lower values mean more significant query modification.",
        "unit": "ratio (0-1)",
        "good_range": "0.3 - 0.8"
    }
}


async def get_metric_explanations(request: Request):
    """Get explanations for all tracked metrics."""
    return JSONResponse(METRIC_EXPLANATIONS)


async def chunks_analytics_page(request: Request):
    """Render the chunks analytics dashboard."""
    return templates.TemplateResponse(
        "chunks_analytics.html",
        {"request": request}
    )


async def get_chunks_analytics(request: Request):
    """Get comprehensive chunks analytics from performance data."""
    try:
        limit = int(request.query_params.get("limit", 500))
        tracker = get_performance_tracker(log_dir=settings.performance_log_dir)
        records = tracker.get_recent_records(limit=limit)
        
        # Aggregate chunk statistics
        all_chunk_ids = set()
        chunk_usage_count = defaultdict(int)  # chunk_id -> times used
        chunk_sources = {}  # chunk_id -> chunk source info
        chunks_per_query = []
        kb_usage = defaultdict(int)  # kb_name -> times used
        total_queries = len(records)
        total_chunks_retrieved = 0
        
        # Session-based chunk grouping
        session_chunks = []  # List of { session_id, query, timestamp, chunks[] }
        
        for record in records:
            retrieval = record.get("retrieval", {})
            session = record.get("session", {})
            query = record.get("query", {})
            
            chunks_retrieved = retrieval.get("chunks_retrieved", 0)
            total_chunks_retrieved += chunks_retrieved
            chunks_per_query.append(chunks_retrieved)
            
            chunk_sources_list = retrieval.get("chunk_sources", [])
            session_chunk_data = {
                "session_id": session.get("session_id", ""),
                "request_id": session.get("request_id", ""),
                "query": query.get("original_query", ""),
                "enhanced_query": query.get("enhanced_query", ""),
                "timestamp": session.get("timestamp", ""),
                "channel": query.get("channel", ""),
                "chunks": []
            }
            
            for chunk in chunk_sources_list:
                chunk_id = chunk.get("chunk_id")
                if chunk_id:
                    all_chunk_ids.add(chunk_id)
                    chunk_usage_count[chunk_id] += 1
                    
                    # Store chunk source info (with latest data)
                    chunk_sources[chunk_id] = {
                        "chunk_id": chunk_id,
                        "kb_name": chunk.get("kb_name", "Unknown"),
                        "kb_id": chunk.get("kb_id", ""),
                        "file_path": chunk.get("file_path", ""),
                        "text_preview": chunk.get("text_preview", ""),
                        "similarity": chunk.get("similarity", 0)
                    }
                    
                    kb_usage[chunk.get("kb_name", "Unknown")] += 1
                    
                    session_chunk_data["chunks"].append({
                        "chunk_id": chunk_id,
                        "kb_name": chunk.get("kb_name", ""),
                        "similarity": chunk.get("similarity", 0),
                        "text_preview": chunk.get("text_preview", "")
                    })
            
            if session_chunk_data["chunks"]:
                session_chunks.append(session_chunk_data)
        
        # Calculate statistics
        avg_chunks_per_query = sum(chunks_per_query) / len(chunks_per_query) if chunks_per_query else 0
        max_chunks_per_query = max(chunks_per_query) if chunks_per_query else 0
        min_chunks_per_query = min(chunks_per_query) if chunks_per_query else 0
        
        # Get top used chunks (sorted by usage count)
        top_chunks = sorted(
            [(chunk_id, count, chunk_sources.get(chunk_id, {})) 
             for chunk_id, count in chunk_usage_count.items()],
            key=lambda x: x[1],
            reverse=True
        )[:50]  # Top 50 most used chunks
        
        # Sort session chunks by timestamp descending
        session_chunks.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return JSONResponse({
            "statistics": {
                "total_queries": total_queries,
                "unique_chunks_used": len(all_chunk_ids),
                "total_chunks_retrieved": total_chunks_retrieved,
                "avg_chunks_per_query": round(avg_chunks_per_query, 2),
                "max_chunks_per_query": max_chunks_per_query,
                "min_chunks_per_query": min_chunks_per_query,
            },
            "kb_usage": dict(kb_usage),
            "top_chunks": [
                {
                    "chunk_id": chunk_id,
                    "usage_count": count,
                    **info
                }
                for chunk_id, count, info in top_chunks
            ],
            "session_chunks": session_chunks[:100],  # Last 100 sessions
            "all_chunk_ids": list(all_chunk_ids)
        })
    except Exception as e:
        logger.error(f"Error fetching chunks analytics: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_chunk_details(request: Request):
    """Fetch full chunk details from database by chunk IDs."""
    try:
        # Get chunk IDs from query params or request body
        chunk_ids_param = request.query_params.get("chunk_ids", "")
        chunk_ids = [cid.strip() for cid in chunk_ids_param.split(",") if cid.strip()]
        
        if not chunk_ids:
            return JSONResponse({"error": "No chunk_ids provided", "chunks": []}, status_code=400)
        
        # Limit to prevent too large queries
        chunk_ids = chunk_ids[:50]
        
        db = SessionLocal()
        try:
            # Query chunks from database with knowledge base info
            chunks_data = []
            for chunk_id in chunk_ids:
                try:
                    result = db.execute(
                        select(Embedding, KnowledgeBase.name)
                        .join(KnowledgeBase, Embedding.kb_id == KnowledgeBase.id)
                        .where(Embedding.id == chunk_id)
                    ).first()
                    
                    if result:
                        embedding, kb_name = result
                        chunks_data.append({
                            "chunk_id": str(embedding.id),
                            "kb_id": str(embedding.kb_id),
                            "kb_name": kb_name,
                            "chunk_text": embedding.chunk_text,
                            "chunk_metadata": embedding.chunk_metadata or {},
                            "created_at": embedding.created_at.isoformat() if embedding.created_at else None,
                            "text_length": len(embedding.chunk_text) if embedding.chunk_text else 0
                        })
                except Exception as e:
                    logger.warning(f"Error fetching chunk {chunk_id}: {e}")
                    continue
                    
            return JSONResponse({
                "chunks": chunks_data,
                "found": len(chunks_data),
                "requested": len(chunk_ids)
            })
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error fetching chunk details: {e}", exc_info=True)
        return JSONResponse({"error": str(e), "chunks": []}, status_code=500)


async def get_chunk_by_id(request: Request):
    """Get a single chunk's full details by ID."""
    try:
        chunk_id = request.path_params.get("chunk_id")
        if not chunk_id:
            return JSONResponse({"error": "No chunk_id provided"}, status_code=400)
        
        db = SessionLocal()
        try:
            result = db.execute(
                select(Embedding, KnowledgeBase.name, KnowledgeBase.description)
                .join(KnowledgeBase, Embedding.kb_id == KnowledgeBase.id)
                .where(Embedding.id == chunk_id)
            ).first()
            
            if not result:
                return JSONResponse({"error": "Chunk not found"}, status_code=404)
            
            embedding, kb_name, kb_description = result
            return JSONResponse({
                "chunk_id": str(embedding.id),
                "kb_id": str(embedding.kb_id),
                "kb_name": kb_name,
                "kb_description": kb_description,
                "chunk_text": embedding.chunk_text,
                "chunk_metadata": embedding.chunk_metadata or {},
                "created_at": embedding.created_at.isoformat() if embedding.created_at else None,
                "text_length": len(embedding.chunk_text) if embedding.chunk_text else 0
            })
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error fetching chunk: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


performance_routes = [
    Route("/performance", endpoint=performance_page, methods=["GET"]),
    Route("/performance/chunks", endpoint=chunks_analytics_page, methods=["GET"]),
    Route("/api/performance/records", endpoint=get_performance_records, methods=["GET"]),
    Route("/api/performance/stats", endpoint=get_aggregate_stats, methods=["GET"]),
    Route("/api/performance/slow", endpoint=get_slow_queries, methods=["GET"]),
    Route("/api/performance/failed", endpoint=get_failed_queries, methods=["GET"]),
    Route("/api/performance/metrics-info", endpoint=get_metric_explanations, methods=["GET"]),
    Route("/api/performance/chunks-analytics", endpoint=get_chunks_analytics, methods=["GET"]),
    Route("/api/performance/chunk-details", endpoint=get_chunk_details, methods=["GET"]),
    Route("/api/performance/chunk/{chunk_id}", endpoint=get_chunk_by_id, methods=["GET"]),
]

