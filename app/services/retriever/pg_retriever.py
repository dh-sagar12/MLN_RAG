import time
import logging
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List, Optional, TYPE_CHECKING

from llama_index.core.schema import NodeWithScore, TextNode

from app.services.config_service import ConfigService

if TYPE_CHECKING:
    from app.services.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class PostgresRetriever(BaseRetriever):
    """Custom retriever for PostgreSQL with pgvector."""

    def __init__(
        self,
        db: Session,
        embed_model: OpenAIEmbedding,
        similarity_threshold: float,
        performance_tracker: Optional["PerformanceTracker"] = None,
        request_id: Optional[str] = None,
        metadata_filters: Optional[MetadataFilters] = None,
    ):
        self.db = db
        self.embed_model = embed_model
        self.top_k = (
            ConfigService.get_retriever_config(db).get("vector", {}).get("top_k")
        )
        self.similarity_threshold = similarity_threshold
        self.performance_tracker = performance_tracker
        self.request_id = request_id
        self.ef_search = ConfigService.get_retriever_config(db)["vector"]["ef_search"]
        self.metadata_filters = metadata_filters
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given a query."""
        query_text = query_bundle.query_str
        retrieval_start = time.perf_counter()

        # Embed query with timing
        embedding_start = time.perf_counter()
        query_embedding = self.embed_model.get_text_embedding(text=query_text)
        embedding_time_ms = (time.perf_counter() - embedding_start) * 1000

        # PERFORMANCE TRACKING: Record embedding generation
        if self.performance_tracker and self.request_id:
            self.performance_tracker._record_timing(
                self.request_id, "embedding_generation", embedding_time_ms
            )
            # Add embedding metadata
            self.performance_tracker.add_metadata(
                self.request_id, "embedding_dimension", len(query_embedding)
            )

        query_vector = list(query_embedding)
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"

        # SQL for cosine similarity with timing
        vector_search_start = time.perf_counter()

        metadata_sql, metadata_params = self._build_metadata_filter_sql()
        sql = text(
            f"""
            SET hnsw.ef_search = :ef_search;
            SELECT 
                e.id,
                e.kb_id,
                e.chunk_text,
                e.chunk_metadata,
                kb.name as kb_name,
                1 - (e.embedding <=> cast(:query_vector AS vector)) as similarity
            FROM embeddings e
            JOIN knowledge_bases kb ON e.kb_id = kb.id
            WHERE 1 - (e.embedding <=> cast(:query_vector AS vector)) > :similarity_threshold 
            {metadata_sql}
            ORDER BY e.embedding <=> cast(:query_vector AS vector)
            LIMIT :top_k
        """
        )  # NOTE: change similarity threshold later to required

        result = self.db.execute(
            sql,
            {
                "query_vector": vector_str,
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
                "ef_search": self.ef_search,
                **metadata_params,
            }
        )
        rows = result.fetchall()
        
        compiled = sql.compile(
            dialect=self.db.bind.dialect,
            compile_kwargs={"literal_binds": True},
        )
        print(f"Actual SQL:\n{str(compiled)}")

        logger.info(
            f"Vector search retrieved {len(rows)} nodes with similarity threshold {self.similarity_threshold}"
        )

        # PERFORMANCE TRACKING: Record vector search and retrieval total
        vector_search_time_ms = (time.perf_counter() - vector_search_start) * 1000
        retrieval_total_time_ms = (time.perf_counter() - retrieval_start) * 1000
        if self.performance_tracker and self.request_id:
            self.performance_tracker._record_timing(
                self.request_id, "vector_search", vector_search_time_ms
            )
            self.performance_tracker._record_timing(
                self.request_id, "retrieval_total", retrieval_total_time_ms
            )
            # Add search metadata
            self.performance_tracker.add_metadata(
                self.request_id, key="hnsw_ef_search", value=self.ef_search
            )
            self.performance_tracker.add_metadata(
                self.request_id, key="rows_returned", value=len(rows)
            )
            self.performance_tracker.add_metadata(
                self.request_id, key="metadata_filters", value=self.metadata_filters
            )

        nodes = []
        for row in rows:
            # Create TextNode
            node = TextNode(
                text=row.chunk_text,
                metadata={
                    "kb_id": str(row.kb_id),
                    "kb_name": row.kb_name,
                    "embedding_id": str(row.id),  # Add embedding ID for tracking
                    **(row.chunk_metadata or {}),
                },
            )
            nodes.append(
                NodeWithScore(
                    node=node,
                    score=float(row.similarity),
                ),
            )

        logger.debug(
            f"Retrieval completed - Embedding: {embedding_time_ms:.2f}ms, "
            f"Vector Search: {vector_search_time_ms:.2f}ms, "
            f"Total: {retrieval_total_time_ms:.2f}ms, "
            f"Rows: {len(rows)}"
        )

        return nodes

    def _build_metadata_filter_sql(self):
        """
        Convert LlamaIndex MetadataFilters → SQL WHERE conditions for JSONB metadata.
        Supports:
            - ExactMatchFilter
            - ContainsFilter
            - NumericFilter (gt, gte, lt, lte)
        """
        if not self.metadata_filters:
            return "", {}

        conditions = []
        params = {}

        for idx, f in enumerate(self.metadata_filters.filters):
            key = f.key
            param_key = f"mf_{idx}"

            # Exact match
            if f.operator == "eq" and f.value:
                conditions.append(f"e.chunk_metadata ->> :mf_key_{idx} = :{param_key}")
                params[f"mf_key_{idx}"] = key
                params[param_key] = str(f.value)

            # Contains (text)
            elif f.operator == "contains" and f.value:
                conditions.append(
                    f"e.chunk_metadata ->> :mf_key_{idx} ILIKE :{param_key}"
                )
                params[f"mf_key_{idx}"] = key
                params[param_key] = f"%{f.value}%"

            # Numeric comparisons
            elif f.operator in ["gt", "gte", "lt", "lte"] and f.value:
                op = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[f.operator]
                conditions.append(
                    f"(e.chunk_metadata ->> :mf_key_{idx})::numeric {op} :{param_key}"
                )
                params[f"mf_key_{idx}"] = key
                params[param_key] = f.value

            elif f.operator == "list_intersects" and f.value:
                conditions.append(
                    f"(e.chunk_metadata -> :mf_key_{idx})::jsonb ?| :{param_key}"
                )
                params[f"mf_key_{idx}"] = key
                params[param_key] = f.value  # must be a Python list
                
        
        logger.info(f"Metdata Filter Conditions: {conditions}")
        logger.info(f"Metdata Filter Parameters: {params}")

        if conditions:
            return " AND " + " AND ".join(conditions), params

        return "", {}
