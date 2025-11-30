"""Performance tracking routes."""

import json
import logging
from pathlib import Path
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.templating import Jinja2Templates

from app.config import settings
from app.services.performance_tracker import get_performance_tracker

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


performance_routes = [
    Route("/performance", endpoint=performance_page, methods=["GET"]),
    Route("/api/performance/records", endpoint=get_performance_records, methods=["GET"]),
    Route("/api/performance/stats", endpoint=get_aggregate_stats, methods=["GET"]),
    Route("/api/performance/slow", endpoint=get_slow_queries, methods=["GET"]),
    Route("/api/performance/failed", endpoint=get_failed_queries, methods=["GET"]),
    Route("/api/performance/metrics-info", endpoint=get_metric_explanations, methods=["GET"]),
]

