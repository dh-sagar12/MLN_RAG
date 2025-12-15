"""AWS Lambda handler for reranking documents using SentenceTransformer.

This Lambda function receives a query and documents, then uses a reranking
model (BAAI/bge-reranker-large) to score document relevance.

Deploy this function with the following configuration:
- Memory: 1024 MB minimum (2048 MB recommended for better performance)
- Timeout: 60 seconds
- Runtime: Python 3.11
- Architecture: x86_64 or arm64

Required layers or dependencies:
- torch
- transformers
- sentence-transformers
"""

import json
import logging
from typing import Any, Dict, List

# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global model cache to keep model warm between invocations
_reranker_model = None


def get_reranker_model():
    """Load and cache the reranking model.
    
    Uses lazy loading to initialize the model only on first invocation,
    keeping it in memory for subsequent requests (warm starts).
    """
    global _reranker_model
    
    if _reranker_model is None:
        logger.info("Loading reranker model...")
        from sentence_transformers import CrossEncoder
        
        # You can change this to other models like:
        # - "BAAI/bge-reranker-base" (smaller, faster)
        # - "cross-encoder/ms-marco-MiniLM-L-6-v2" (very fast, good for latency)
        _reranker_model = CrossEncoder(
            "BAAI/bge-reranker-large",
            max_length=512,
        )
        logger.info("Reranker model loaded successfully")
    
    return _reranker_model


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle reranking request.
    
    Expected input format:
    {
        "query": "user query string",
        "documents": [
            {"index": 0, "text": "document text 1", "original_score": 0.9},
            {"index": 1, "text": "document text 2", "original_score": 0.85},
            ...
        ],
        "top_n": 5
    }
    
    Output format:
    {
        "scores": [
            {"index": 0, "score": 0.95},
            {"index": 2, "score": 0.87},
            ...
        ]
    }
    
    Args:
        event: Lambda event containing query and documents.
        context: Lambda context (unused).
        
    Returns:
        Dictionary with reranked scores.
    """
    try:
        # Parse input
        query = event.get("query", "")
        documents = event.get("documents", [])
        top_n = event.get("top_n", 5)
        
        if not query:
            return {"errorMessage": "No query provided"}
        
        if not documents:
            return {"scores": []}
        
        logger.info(f"Reranking {len(documents)} documents for query: {query[:100]}...")
        
        # Load model
        model = get_reranker_model()
        
        # Prepare query-document pairs for reranking
        pairs = [(query, doc["text"]) for doc in documents]
        print("pairs------------>", pairs)
        
        # Get reranking scores
        scores = model.predict(pairs)
        
        print("scores------------>", scores)
        
        # Combine scores with original indices
        scored_docs: List[Dict[str, Any]] = []
        for i, doc in enumerate(documents):
            scored_docs.append({
                "index": doc["index"],
                "score": float(scores[i]),
            })
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top_n results
        result_scores = scored_docs[:top_n]
        
        logger.info(f"Reranking complete. Top score: {result_scores[0]['score']:.4f}")
        
        return {"scores": result_scores}
        
    except Exception as e:
        logger.error(f"Error during reranking: {str(e)}", exc_info=True)
        return {"errorMessage": str(e)}

