


import logging
from typing import List
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from sqlalchemy import text
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25 keyword-based retriever for sparse search."""

    def __init__(
        self,
        db: Session,
        top_k: int = 50,
        language: str = "english",
    ):
        self.db = db
        self.top_k = top_k
        self.language = language
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve using PostgreSQL full-text search (BM25-like)."""
        query_text = query_bundle.query_str

        # Use PostgreSQL ts_rank for BM25-like ranking
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
                AND kb.is_active = true
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

        logger.info(f"BM25 search retrieved {len(nodes)} nodes")
        return nodes

