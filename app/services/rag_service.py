"""RAG service for querying across knowledge bases."""

import logging
from typing import List, Dict, Any, Optional, AsyncIterator
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from sqlalchemy.sql import text
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from app.config import settings
import asyncio
from llama_index.core.llms import ChatMessage as LlamaChatMessage

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG querying."""

    def __init__(self, db: Session):
        self.db = db
        if settings.openai_api_key:
            # Clear proxy environment variables to avoid OpenAI client initialization issues
            import os

            # proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
            # saved_proxies = {}
            # for var in proxy_vars:
            #     if var in os.environ:
            #         saved_proxies[var] = os.environ.pop(var)

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
                    f"OpenAI clients initialized (embedding: {settings.openai_embedding_model}, LLM: {settings.openai_llm_model})"
                )
            except Exception as e:
                logger.error(f"Error initializing OpenAI clients: {e}", exc_info=True)
                raise
            finally:
                # Restore proxy environment variables
                # for var, value in saved_proxies.items():
                #     os.environ[var] = value
                pass
        else:
            self.embed_model = None
            self.llm = None

    def get_enhanced_query(self, query_text, chat_history):
        SYSTEM_PROMPT = """
            You generate precise semantic-search queries for a vector database.

            Your goal:
            - Use conversation history ONLY to resolve context and references.
            - Base the enhanced query STRICTLY on the user's latest question.
            - Do NOT add information the user did not ask for.
            - Do NOT answer the question.
            - Output exactly one concise semantic-search query.
            - Keep it short, factual, and focused on the latest request.
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": combined_input},
        ]

        chat_messages = [
            LlamaChatMessage(role=m["role"], content=m["content"])
            for m in messages
        ]

        response = self.llm.chat(chat_messages)
        return str(response.message.content).strip()

        

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Query across all knowledge bases.

        Args:
            query_text: User query
            top_k: Number of chunks to retrieve

        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"Processing query: '{query_text[:50]}...' (top_k={top_k})")

        if not self.embed_model:
            logger.warning("OpenAI API key not configured, returning error response")
            return {
                "answer": "OpenAI API key not configured.",
                "sources": [],
                "kbs_used": [],
                "chunks": [],
            }

        # Enchance the user new query with the past context for better semectic search

        enhanced_query = self.get_enhanced_query(
            query_text=query_text,
            chat_history=chat_history,
        )
        print(enhanced_query, 'enhanced_query')

        # Embed query (sync call, but OpenAI API is async internally)
        logger.debug("Generating query embedding...")
        query_embedding = self.embed_model.get_text_embedding(text=enhanced_query)
        logger.debug("Query embedding generated")

        # Convert to list for pgvector
        query_vector = list(query_embedding)

        # Perform similarity search across all knowledge bases
        # Using cosine distance (1 - cosine similarity)
        # Convert list to string format for pgvector
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"

        sql = text(
            """
            SELECT 
                e.id,
                e.kb_id,
                e.chunk_text,
                e.chunk_metadata,
                kb.name as kb_name,
                1 - (e.embedding <=> cast(:query_vector AS vector)) as similarity
            FROM embeddings e
            JOIN knowledge_bases kb ON e.kb_id = kb.id
            ORDER BY e.embedding <=> cast(:query_vector AS vector)
            LIMIT :top_k
        """
        )

        # NOTE: <=> : this operator in above query indicate the cosine distance in pgvector

        logger.debug(f"Executing vector similarity search (top_k={top_k})")
        result = self.db.execute(sql, {"query_vector": vector_str, "top_k": top_k})
        rows = result.fetchall()
        logger.info(f"Found {len(rows)} relevant chunks")

        if not rows:
            logger.warning("No relevant chunks found for query")
            return {
                "answer": "No relevant information found in the knowledge bases.",
                "sources": [],
                "kbs_used": [],
                "chunks": [],
            }

        # Extract chunks and metadata
        chunks = []
        kbs_used = set()
        sources = []

        for row in rows:
            chunks.append(
                {
                    "text": row.chunk_text,
                    "kb_id": str(row.kb_id),
                    "kb_name": row.kb_name,
                    "similarity": float(row.similarity),
                    "metadata": row.chunk_metadata or {},
                }
            )
            kbs_used.add(row.kb_name)
            if row.chunk_metadata and "file_path" in row.chunk_metadata:
                sources.append(row.chunk_metadata["file_path"])

        # Build context from chunks
        context = "\n\n".join(
            [f"[Source: {chunk['kb_name']}]\n{chunk['text']}" for chunk in chunks]
        )

        # Generate answer using LLM with chat history
        logger.debug("Generating answer using LLM...")

        # Build messages for chat completion
        messages = []

        # Add system message
        system_message = f"""
        You are an AI assistant for the reservations and sales team of Mountain Lodges of Nepal, a premium Himalayan hospitality and travel company.

        Your job is to draft highly accurate, polite, and concise replies to guest messages
        based ONLY on the information provided in the CONTEXT and THREAD sections below.

        STRICT RULES:

        - Do not invent prices, availability, or dates.
        - If prices or availability are needed but not given in CONTEXT, say that the team
        will confirm those details and phrase the reply accordingly.
        - Prefer the most recent information and active policies. If there are multiple conflicting
        snippets, use the one with the latest effective date.
        - Respect the channel:
            - Email: complete sentences, slightly more formal.
            - WhatsApp: friendly, shorter paragraphs, but still professional.
        - Do not include internal metadata or technical labels in your answer.
        - Never quote obviously outdated weather or pricing (e.g., from 2020 or a past season).

        If the CONTEXT is clearly incomplete for a safe answer, explicitly suggest what the
        human agent should double-check (e.g., "Please confirm availability for these dates
        before finalizing this email."), but still draft the rest of the message.

        You are an AI assistant for the reservations and sales team of Mountain Lodges of Nepal,
        a premium Himalayan hospitality and travel company.

        Your job is to draft highly accurate, polite, and concise replies to guest messages
        based ONLY on the information provided in the CONTEXT and THREAD sections below.

        STRICT RULES:

        - Do not invent prices, availability, or dates.
        - If prices or availability are needed but not given in CONTEXT, say that the team
        will confirm those details and phrase the reply accordingly.
        - Prefer the most recent information and active policies. If there are multiple conflicting
        snippets, use the one with the latest effective date.
        - Respect the channel:
            - Email: complete sentences, slightly more formal.
            - WhatsApp: friendly, shorter paragraphs, but still professional.
        - Do not include internal metadata or technical labels in your answer.
        - Never quote obviously outdated weather or pricing (e.g., from 2020 or a past season).

        If the CONTEXT is clearly incomplete for a safe answer, explicitly suggest what the
        human agent should double-check (e.g., "Please confirm availability for these dates
        before finalizing this email."), but still draft the rest of the message
        """
        messages.append({"role": "system", "content": system_message})

        # Add chat history if provided (last k messages)
        if chat_history:
            logger.debug(f"Including {len(chat_history)} previous messages in context")
            for msg in chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Add context and current question
        context_message = f"""Based on the following context from knowledge bases and past conversations, answer the user's question.
        Context:
        {context}
        Question: {query_text}
        Answer:"""
        messages.append({"role": "user", "content": context_message})

        if self.llm:

            chat_messages = [
                LlamaChatMessage(role=msg["role"], content=msg["content"])
                for msg in messages
            ]
            response = self.llm.chat(chat_messages)
            answer_text = str(response.message.content).strip()
            logger.info("LLM answer generated successfully")
        else:
            logger.warning("LLM not configured, returning context chunks")
            answer_text = (
                "LLM not configured. Here are the relevant chunks:\n\n" + context
            )

        return {
            "answer": answer_text,
            "sources": list(set(sources)),
            "kbs_used": list(kbs_used),
            "chunks": chunks,
        }

    async def stream_query(self, query_text: str, top_k: int = 5) -> AsyncIterator[str]:
        """Stream query response.

        Args:
            query_text: User query
            top_k: Number of chunks to retrieve

        Yields:
            Chunks of the response
        """
        # First get the context (same as query)
        result = await asyncio.to_thread(self.query, query_text, top_k)

        # For streaming, we'll yield the answer in chunks
        # In a real implementation, you'd use the LLM's streaming API
        answer = result["answer"]
        chunk_size = 50
        for i in range(0, len(answer), chunk_size):
            yield answer[i : i + chunk_size]
