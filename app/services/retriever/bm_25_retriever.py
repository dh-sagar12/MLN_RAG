import time
import logging
from typing import List, Optional, TYPE_CHECKING
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from sqlalchemy import text
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from app.services.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25 keyword-based retriever for sparse search."""

    def __init__(
        self,
        db: Session,
        top_k: int = 50,
        language: str = "english",
        performance_tracker: Optional["PerformanceTracker"] = None,
        request_id: Optional[str] = None,
    ):
        self.db = db
        self.top_k = top_k
        self.language = language
        self.performance_tracker = performance_tracker
        self.request_id = request_id
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve using PostgreSQL full-text search (BM25-like)."""
        query_text = query_bundle.query_str
        retrieval_start = time.perf_counter()

        # Use PostgreSQL ts_rank for BM25-like ranking
        bm25_search_start = time.perf_counter()
        
        sql = text(
            """
            SELECT 
                e.id,
                e.kb_id,
                e.chunk_text,
                e.chunk_metadata,
                kb.name as kb_name,
                ts_rank_cd(
                    to_tsvector(:language, e.chunk_text),
                    plainto_tsquery(:language, :query_text)
                ) as rank
            FROM embeddings e
            JOIN knowledge_bases kb ON e.kb_id = kb.id
            WHERE to_tsvector(:language, e.chunk_text) @@ plainto_tsquery(:language, :query_text)
            ORDER BY rank DESC
            LIMIT :top_k
            """
        )

        result = self.db.execute(
            sql,
            {
                "query_text": query_text,
                "top_k": self.top_k,
                "language": self.language,
            },
        )
        rows = result.fetchall()
        
        bm25_search_time_ms = (time.perf_counter() - bm25_search_start) * 1000
        retrieval_total_time_ms = (time.perf_counter() - retrieval_start) * 1000
        
        # Record performance metrics
        if self.performance_tracker and self.request_id:
            self.performance_tracker._record_timing(
                self.request_id, "bm25_search", bm25_search_time_ms
            )
            self.performance_tracker._record_timing(
                self.request_id, "retrieval_total", retrieval_total_time_ms
            )
            # Add BM25-specific metadata
            self.performance_tracker.add_metadata(
                self.request_id, "bm25_language", self.language
            )
            self.performance_tracker.add_metadata(
                self.request_id, "bm25_top_k", self.top_k
            )
            self.performance_tracker.add_metadata(
                self.request_id, "bm25_rows_returned", len(rows)
            )
            self.performance_tracker.add_metadata(
                self.request_id, "retrieval_type", "bm25"
            )

        nodes = []
        for row in rows:
            node = TextNode(
                text=row.chunk_text,
                metadata={
                    "kb_id": str(row.kb_id),
                    "kb_name": row.kb_name,
                    "embedding_id": str(row.id),
                    **(row.chunk_metadata or {}),
                },
            )
            nodes.append(
                NodeWithScore(
                    node=node,
                    score=float(row.rank) if row.rank else 0.0,
                )
            )

        logger.debug(
            f"BM25 retrieval completed - Search: {bm25_search_time_ms:.2f}ms, "
            f"Total: {retrieval_total_time_ms:.2f}ms, "
            f"Rows: {len(rows)}"
        )
        logger.info(f"BM25 search retrieved {len(nodes)} nodes")
        return nodes

