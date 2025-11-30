
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List
from llama_index.core.schema import NodeWithScore, TextNode

class PostgresRetriever(BaseRetriever):
    """Custom retriever for PostgreSQL with pgvector."""

    def __init__(
        self,
        db: Session,
        embed_model: OpenAIEmbedding,
        top_k: int = 5,
        similarity_threshold: float = 0.75,
    ):
        self.db = db
        self.embed_model = embed_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given a query."""
        query_text = query_bundle.query_str

        # Embed query
        query_embedding = self.embed_model.get_text_embedding(text=query_text)
        query_vector = list(query_embedding)
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"

        # SQL for cosine similarity
        
        sql = text(
            """
            SET hnsw.ef_search = 256;
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
            ORDER BY e.embedding <=> cast(:query_vector AS vector)
            LIMIT :top_k
        """
        ) #NOTE: change similarity threshold later to required

        result = self.db.execute(
            sql,
            {
                "query_vector": vector_str,
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
            }
        )
        rows = result.fetchall()

        nodes = []
        for row in rows:
            # Create TextNode
            node = TextNode(
                text=row.chunk_text,
                metadata={
                    "kb_id": str(row.kb_id),
                    "kb_name": row.kb_name,
                    **(row.chunk_metadata or {}),
                },
            )
            nodes.append(
                NodeWithScore(
                    node=node,
                    score=float(row.similarity),
                ),
            )

        return nodes

