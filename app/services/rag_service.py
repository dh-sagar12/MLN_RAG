"""RAG service for querying across knowledge bases."""

import logging
from typing import List, Dict, Any, Optional, AsyncIterator
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
    get_response_synthesizer,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llms import ChatMessage as LlamaChatMessage
from app.config import settings
import asyncio
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank

logger = logging.getLogger(__name__)


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


class RAGService:
    """Service for RAG querying using LlamaIndex."""

    def __init__(self, db: Session):
        self.db = db
        self.embed_model = None
        self.llm = None

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
                    f"OpenAI clients initialized (embedding: {settings.openai_embedding_model}",
                    f"LLM: {settings.openai_llm_model}",
                    f"Temperature: {settings.temperature})",
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

        SYSTEM_PROMPT = """
            You generate precise semantic-search queries for a vector database.

            Your goal:
            - Use conversation history ONLY to resolve context and references.
            - Base the enhanced query STRICTLY on the user's latest question.
            - Do NOT add information the user did not ask for.
            - Do NOT answer the question.
            - Output exactly one concise semantic-search query.
            - Keep it short, factual, and focused on the latest request.
            - Give reponse on Markdown Format with proper spacing and formatting.
        """

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
                """Your role is to write friendly, concise, and accurate WhatsApp messages based only on the information in the CONTEXT and THREAD.

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
    ) -> Dict[str, Any]:
        """Query across all knowledge bases.

        Args:
            query_text: User query
            top_k: Number of chunks to retrieve
            similarity_threshold: Similarity threshold for retrieving chunks
            channel: Output channel style (email or whatsapp)

        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"Processing query: '{query_text[:50]}...' (top_k={top_k})")

        if not self.embed_model or not self.llm:
            logger.warning("OpenAI API key not configured, returning error response")
            return {
                "answer": "OpenAI API key not configured.",
                "sources": [],
                "kbs_used": [],
                "chunks": [],
            }

        # Enhance query
        enhanced_query = self.get_enhanced_query(
            query_text=query_text,
            chat_history=chat_history,
        )
        logger.info(f"Enhanced query: {enhanced_query}")

        # Initialize Retriever
        retriever = PostgresRetriever(
            db=self.db,
            embed_model=self.embed_model,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        # Build Prompt Template
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

        # Configure Response Synthesizer
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            text_qa_template=qa_template,
            response_mode="compact",
        )

        # Configure Query Engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        # Execute Query
        response = query_engine.query(enhanced_query)

        # Extract response and metadata
        answer_text = str(response)

        chunks = []
        kbs_used = set()
        sources = []

        for node_with_score in response.source_nodes:
            node = node_with_score.node
            score = node_with_score.score

            kb_name = node.metadata.get("kb_name", "Unknown")

            chunks.append(
                {
                    "text": node.get_content(),
                    "kb_id": node.metadata.get("kb_id"),
                    "kb_name": kb_name,
                    "similarity": score,
                    "metadata": node.metadata,
                }
            )
            kbs_used.add(kb_name)
            if "file_path" in node.metadata:
                sources.append(node.metadata["file_path"])

        return {
            "answer": answer_text,
            "sources": list(set(sources)),
            "kbs_used": list(kbs_used),
            "chunks": chunks,
        }

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
