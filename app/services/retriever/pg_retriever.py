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
        retrieval_start = time.perf_counter()

        query_texts = query_bundle.custom_embedding_strs
        if not query_texts:
            return []

        metadata_sql, metadata_params = self._build_metadata_filter_sql()

        merged_results: dict[str, NodeWithScore] = {}

        embedding_start = time.perf_counter()
        query_embeddings = self.embed_model.get_text_embedding_batch(
            texts=query_texts,
        )

        embedding_time_ms = (time.perf_counter() - embedding_start) * 1000
        if self.performance_tracker and self.request_id:
            self.performance_tracker._record_timing(
                self.request_id,
                "embedding_generation",
                embedding_time_ms,
            )

        for query_embedding in query_embeddings:

            vector_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # 2. Vector search for this query
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
            """,
            )

            result = self.db.execute(
                sql,
                {
                    "query_vector": vector_str,
                    "top_k": self.top_k,
                    "similarity_threshold": self.similarity_threshold,
                    "ef_search": self.ef_search,
                    **metadata_params,
                },
            )

            rows = result.fetchall()

            # 3. Merge results (keep best score per chunk)
            for row in rows:
                embedding_id = str(row.id)
                score = float(row.similarity)

                if (
                    embedding_id not in merged_results
                    or score > merged_results[embedding_id].score
                ):
                    node = TextNode(
                        text=row.chunk_text,
                        metadata={
                            "kb_id": str(row.kb_id),
                            "kb_name": row.kb_name,
                            "embedding_id": embedding_id,
                            **(row.chunk_metadata or {}),
                        },
                    )
                    merged_results[embedding_id] = NodeWithScore(
                        node=node,
                        score=score,
                    )

        # 4. Final ranking
        final_nodes = sorted(
            merged_results.values(),
            key=lambda n: n.score,
            reverse=True,
        )
        
        # print(final_nodes, 'final nodes')

        retrieval_total_time_ms = (time.perf_counter() - retrieval_start) * 1000

        if self.performance_tracker and self.request_id:
            self.performance_tracker._record_timing(
                self.request_id,
                "retrieval_total",
                retrieval_total_time_ms,
            )
            self.performance_tracker.add_metadata(
                self.request_id,
                "queries_processed",
                len(query_texts),
            )
            self.performance_tracker.add_metadata(
                self.request_id,
                "final_nodes",
                len(final_nodes),
            )

        return final_nodes

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
