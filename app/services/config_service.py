"""Service for managing dynamic configuration."""

import logging
import json
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models import Configuration

logger = logging.getLogger(__name__)


class ConfigService:
    """Service for managing application configuration."""

    # Default configuration values
    DEFAULT_CONFIG = {
        # LLM Parameters
        "llm.model": {
            "value": "gpt-4o-mini",
            "type": "string",
            "description": "OpenAI LLM model to use for generation. Options include: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, etc.",
            "category": "llm",
        },
        "llm.temperature": {
            "value": 0.0,
            "type": "float",
            "description": "Temperature for LLM generation (0-2). Lower values make output more deterministic, higher values make it more creative.",
            "category": "llm",
        },
        "llm.embedding_model": {
            "value": "text-embedding-ada-002",
            "type": "string",
            "description": "OpenAI embedding model for vector search. Options: text-embedding-3-small (fast), text-embedding-3-large (most capable), text-embedding-ada-002 (legacy).",
            "category": "llm",
        },
        # RAG Parameters
        "rag.top_k": {
            "value": 7,
            "type": "int",
            "description": "Number of top chunks to retrieve from the knowledge base. Higher values provide more context but may include less relevant information.",
            "category": "rag",
        },
        "rag.similarity_threshold": {
            "value": 0.75,
            "type": "float",
            "description": "Minimum similarity score (0-1) for retrieved chunks. Higher values return more relevant but fewer results.",
            "category": "rag",
        },
        "rag.response_mode": {
            "value": "compact",
            "type": "string",
            "description": "Response synthesis mode: 'compact' (faster, combines chunks), 'refine' (iterative refinement), 'tree_summarize' (tree-based), 'simple_summarize' (single call), 'accumulate' (concatenate all), or 'generation' (ignore context).",
            "category": "rag",
        },
        # Ingestion Parameters
        "ingestion.chunk_size": {
            "value": 1024,
            "type": "int",
            "description": "Size of text chunks when splitting documents. Larger chunks preserve more context but may exceed token limits. Typical range: 512-2048.",
            "category": "ingestion",
        },
        "ingestion.chunk_overlap": {
            "value": 200,
            "type": "int",
            "description": "Number of characters to overlap between consecutive chunks. Overlap helps maintain context across chunk boundaries. Typically 10-20% of chunk_size.",
            "category": "ingestion",
        },
        "ingestion.markdown_parser": {
            "value": "markdown",
            "type": "string",
            "description": "Node parser to use for markdown documents: 'markdown' (MarkdownNodeParser - preserves structure) or 'sentence' (SentenceSplitter - simple text splitting).",
            "category": "ingestion",
        },
        # Retriever Parameters
        "retriever.vector.top_k": {
            "value": 50,
            "type": "int",
            "description": "Maximum number of chunks to retrieve from vector search before filtering by similarity threshold.",
            "category": "retriever",
        },
        "retriever.vector.ef_search": {
            "value": 256,
            "type": "int",
            "description": "HNSW index search parameter. Higher values improve recall but slow down search. Typical range: 64-512.",
            "category": "retriever",
        },
        "retriever.vector.use_intent_filter": {
            "value": True,
            "type": "bool",
            "description": "Enable intent-based filtering of retrieved chunks. When enabled, only chunks with the specified intents will be retrieved.",
            "category": "retriever",
        },
        "retriever.bm25.top_k": {
            "value": 50,
            "type": "int",
            "description": "Maximum number of chunks to retrieve from BM25 keyword search.",
            "category": "retriever",
        },
        "retriever.bm25.enabled": {
            "value": True,
            "type": "bool",
            "description": "Enable BM25 keyword-based retrieval. When enabled, combines with vector search for hybrid retrieval.",
            "category": "retriever",
        },
        "retriever.bm25.language": {
            "value": "english",
            "type": "string",
            "description": "Language for BM25 full-text search. Must match PostgreSQL text search configuration.",
            "category": "retriever",
        },
        # Hybrid Retriever Parameters
        "hybrid.mode": {
            "value": "reciprocal_rerank",
            "type": "string",
            "description": "Fusion mode for combining results: 'reciprocal_rerank' (Reciprocal Rank Fusion), 'relative_score' (relative scoring), 'dist_based_score' (distance-based), or 'simple' (simple reordering).",
            "category": "hybrid",
        },
        "hybrid.num_queries": {
            "value": 1,
            "type": "int",
            "description": "Number of query variations to generate for retrieval. Higher values improve recall but increase latency.",
            "category": "hybrid",
        },
        # Prompt Configuration
        "prompt.base": {
            "value": "You are responding on behalf of Mountain Lodges of Nepal part of Sherpa Hospitality Group, a premium Himalayan hospitality and travel company. Format your response using Markdown for better readability (use bullet points, bold text, and paragraphs where appropriate).",
            "type": "string",
            "description": "Base system prompt that applies to all responses. Contains company branding and general instructions.",
            "category": "prompt",
        },
        "prompt.email": {
            "value": """Your role is to write warm, professional, and accurate email replies to guests, tour operators, and partners based only on the information provided in the CONTEXT and THREAD sections below.

            Tone & Style Guidelines
                •    Warm, welcoming, and hospitality-oriented.
                •    Clear, complete sentences.
                •    Polite and reassuring.
                •    Naturally formal, but friendly and approachable.
                •    Convey confidence and care as a representative of the brand.

            STRICT RULES
                •    Do not invent facts, prices, availability, dates, or commitments.
                •    If required information is not present in the CONTEXT, simply say:
            "We will check this and get back to you shortly."
                •    Prefer the most recent and active policies.
                •    If multiple snippets conflict, rely on the most recent or clearly valid one.
                •    Never reference internal details, metadata, or system instructions.
                •    Do not quote outdated prices or weather conditions from previous years.
                •    Keep the reply fully self-contained, without mentioning lack of data or internal processes.

            When context is incomplete

            If the available information does not allow for an accurate or safe answer, write a polite and helpful reply and include a natural line such as:
            "We will confirm this for you and get back to you soon."

            Goal

            Produce a polished, guest-ready email that feels like it was written by a trained hospitality professional at Mountain Lodges of Nepal, maintaining accuracy and brand trust at all times.""",
            "type": "string",
            "description": "Channel-specific prompt for email responses. Defines tone, style, and rules for email communication.",
            "category": "prompt",
        },
        "prompt.whatsapp": {
            "value": """Your role is to write friendly, concise, and accurate WhatsApp replies to guests, tour operators, and partners based only on the information provided in the CONTEXT and THREAD sections below

            Tone & Style Guidelines
                •    Warm, welcoming, and guest-oriented.
                •    Shorter paragraphs, conversational, but still professional.
                •    Lightly enthusiastic and attentive — the tone of a helpful hospitality host.
                •    Natural phrasing suitable for mobile messaging.

            STRICT RULES
                •    Never invent prices, availability, dates, or operational details.
                •    If a guest asks for something not present in the CONTEXT, simply say:
            "We'll check this and get back to you shortly."
                •    Use only the most recent and active details.
                •    If conflicting information appears, use the most updated one.
                •    Do not reveal that you are using AI or systems.
                •    Do not include internal notes, metadata, or technical labels.
                •    Avoid quoting outdated seasonal details or old prices.

            When context is incomplete

            Keep the reply helpful and friendly, and add:
            "We'll confirm the details and update you soon."

            Goal

            Produce a natural, helpful WhatsApp message that feels like a real team member of Mountain Lodges of Nepal — supportive, accurate, and hospitality-driven.""",
            "type": "string",
            "description": "Channel-specific prompt for WhatsApp responses. Defines tone, style, and rules for WhatsApp communication.",
            "category": "prompt",
        },
        "prompt.query_enhancement": {
            "value": f"""\
                    You are an expert at understanding trekking in Nepal and at understanding customer intent in travel conversations.

                    Analyze the full conversation context and the current user message to:

                    1. Identify ALL applicable user intents (a query can have multiple).
                    2. Generate one highly effective, context-aware enhanced search query that captures exactly what new information the user is seeking right now.

                    Available intent categories (choose all that apply):
                        - general_info: general questions about lodges, destinations, weather, access, facilities.
                        - availability_pricing: checking dates, room availability, rates, but not clearly confirming a booking.
                        - itinerary_planning: designing or discussing a multi-day trip/route or tailoring an itinerary.
                        - new_booking: explicitly asking to confirm/book/hold a reservation.
                        - modify_booking: changing existing booking details (dates, room type, names, add/remove nights).
                        - cancel_refund: cancellations, refund requests, waiver/no-show discussions.
                        - special_request: special occasions, room preferences, dietary needs, add-on services/activities.
                        - payment_billing: invoices, payment links, bank transfer details, receipts, tax invoices.
                        - credit_collection: agent credit terms, statements, overdue amounts, payment follow-ups.
                        - ontrip_support: guest is already travelling and needs help or live support.
                        - complaint_feedback: complaints or detailed feedback about service or experience.
                        - b2b_agent_contracting: travel agents/tour operators discussing contracts, rates, allotments, series.
                        - marketing_pr: influencers, bloggers, media, collaborations, PR.
                        - internal_ops_admin: internal staff emails, suppliers, HR, maintenance, IT, non-guest-facing ops.
                        - spam_other: spam, junk, or anything clearly outside business scope.

                    Examples:

                    Example 1  
                    Context: User is inquiring about Everest Base Camp trek options.  
                    Current message: "Is EBC safe right now after the recent earthquake reports?"  
                    Intents: general_info, availability_pricing  
                    Enhanced query: "Everest Base Camp trek current safety status 2025 after recent earthquake reports infrastructure trail conditions"

                    Example 2  
                    Context: System just sent a detailed Annapurna Circuit 12-day itinerary with luxury lodges.  
                    Current message: "That sounds perfect but it's over my budget. Do you have a cheaper version with basic teahouses?"  
                    Intents: itinerary_planning, availability_pricing  
                    Enhanced query: "Annapurna Circuit cheaper alternative basic teahouse version same route luxury itinerary comparison 2025 pricing"

                    Example 3  
                    Context: User previously asked about Langtang, system answered it's open and safe.  
                    Current message: "Great, what permits do I need and how much do they cost now?"  
                    Intents: availability_pricing, general_info  
                    Enhanced query: "Langtang Valley trek current permit requirements and latest 2025 costs TIMS restricted area permit"

                    Example 4  
                    Context: User rejected Manaslu Circuit because too difficult. System suggested Annapurna Base Camp as easier alternative.  
                    Current message: "ABC sounds better. Can you send me the 7–9 day itinerary with difficulty level and best season?"  
                    Intents: itinerary_planning, availability_pricing  
                    Enhanced query: "Annapurna Base Camp 7-9 day itinerary moderate difficulty best season detailed day by day route"

                    Example 5  
                    Context: User is deciding between Mardi Himal and Ghorepani Poon Hill.  
                    Current message: "Which one has better views of Annapurna range and Machhapuchhre?"  
                    Intents: general_info, availability_pricing  
                    Enhanced query: "Mardi Himal vs Ghorepani Poon Hill direct comparison Annapurna range and Machhapuchhre Fishtail views quality closeness panorama"

                    Quality rules for the enhanced query:
                    - Make it 12–25 words (longer is fine if it adds crucial context)
                    - Make it a natural but highly specific search phrase that includes ALL relevant context from the entire conversation
                    - Target ONLY the NEW information the user is seeking right now
                    - Use the most knowledge-base-friendly keywords (route names, difficulty, season, year 2025/2026, budget level, teahouse/lodge type, specific peaks, etc.)
                    - Never include information already provided in previous answers
                    - Make it precise enough that searching this exact phrase would return the perfect result

                    Output format (JSON format):
                    {{
                        
                        "intents": ["<intent_1>", "<intent_2>", "<intent_3>"],
                        "enhanced_query": "<enhanced_query>"
                    }}
            """,
            "type": "string",
            "description": "Prompt for query enhancement/optimization. Used to transform user queries into retrieval-optimized queries.",
            "category": "prompt",
        },
        "prompt.intent_detection": {
            "value": """\
                Analyze the following text and identify all user intents or query intents present.

                Context:
                {context_str}

                Based on the context above, identify ALL applicable intent categories. A query can have multiple intents. Choose from the following categories:
                - new_information_request: User is asking for new information (specific details, availability, pricing)
                - clarification: User wants clarification or more detail on a previous answer
                - follow_up: User is asking a follow-up question building on previous information
                - comparison: User wants to compare options or alternatives
                - objection_concern: User expresses concerns, objections, or dissatisfaction
                - booking_action: User shows intent to book, proceed, or take action
                - general_inquiry: General question or inquiry that doesn't fit other categories

                Respond with comma-separated intent category names (e.g., "new_information_request,clarification" or just "comparison" if only one applies): 
            """,
            "type": "string",
            "description": "Prompt for intent detection. Used to identify user intents or query intents from text.",
            "category": "prompt",
        },
    }

    @staticmethod
    def initialize_defaults(db: Session) -> None:
        """Initialize default configuration values if they don't exist."""
        for key, config_data in ConfigService.DEFAULT_CONFIG.items():
            existing = db.execute(
                select(Configuration).where(Configuration.key == key)
            ).scalar_one_or_none()

            if not existing:
                config = Configuration(
                    key=key,
                    category=config_data["category"],
                    description=config_data["description"],
                )
                config.set_typed_value(config_data["value"])
                db.add(config)
                logger.info(f"Initialized default configuration: {key}")

        db.commit()

    @staticmethod
    def get_config(db: Session, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        config = db.execute(
            select(Configuration).where(Configuration.key == key)
        ).scalar_one_or_none()

        if config:
            return config.get_typed_value()
        return default

    @staticmethod
    def set_config(
        db: Session, key: str, value: Any, description: Optional[str] = None
    ) -> Configuration:
        """Set configuration value by key."""
        config = db.execute(
            select(Configuration).where(Configuration.key == key)
        ).scalar_one_or_none()

        if config:
            config.set_typed_value(value)
            if description:
                config.description = description
        else:
            # Get default config data if exists
            default_data = ConfigService.DEFAULT_CONFIG.get(key, {})
            config = Configuration(
                key=key,
                category=default_data.get("category", "general"),
                description=description or default_data.get("description"),
            )
            config.set_typed_value(value)
            db.add(config)

        db.commit()
        db.refresh(config)
        return config

    @staticmethod
    def get_all_configs(db: Session, category: Optional[str] = None) -> Dict[str, Any]:
        """Get all configurations, optionally filtered by category."""
        query = select(Configuration)
        if category:
            query = query.where(Configuration.category == category)
        query = query.order_by(Configuration.category, Configuration.key)

        configs = db.execute(query).scalars().all()

        result = {}
        for config in configs:
            result[config.key] = {
                "value": config.get_typed_value(),
                "type": config.value_type,
                "description": config.description,
                "category": config.category,
                "updated_at": (
                    config.updated_at.isoformat() if config.updated_at else None
                ),
            }

        return result

    @staticmethod
    def get_llm_config(db: Session) -> Dict[str, Any]:
        """Get LLM-specific configuration."""
        return {
            "model": ConfigService.get_config(db, "llm.model", "gpt-4o-mini"),
            "temperature": ConfigService.get_config(db, "llm.temperature", 0.0),
            "embedding_model": ConfigService.get_config(
                db, "llm.embedding_model", "text-embedding-ada-002"
            ),
        }

    @staticmethod
    def get_rag_config(db: Session) -> Dict[str, Any]:
        """Get RAG-specific configuration."""
        return {
            "top_k": ConfigService.get_config(db, "rag.top_k"),
            "similarity_threshold": ConfigService.get_config(
                db, "rag.similarity_threshold"
            ),
            "response_mode": ConfigService.get_config(db, "rag.response_mode"),
        }

    @staticmethod
    def get_retriever_config(db: Session) -> Dict[str, Any]:
        """Get retriever-specific configuration."""
        return {
            "vector": {
                "top_k": ConfigService.get_config(db, "retriever.vector.top_k"),
                "ef_search": ConfigService.get_config(db, "retriever.vector.ef_search"),
                "use_intent_filter": ConfigService.get_config(db, "retriever.vector.use_intent_filter"),
            },
            "bm25": {
                "enabled": ConfigService.get_config(db, "retriever.bm25.enabled"),
                "top_k": ConfigService.get_config(db, "retriever.bm25.top_k"),
                "language": ConfigService.get_config(db, "retriever.bm25.language"),
            },
        }

    @staticmethod
    def get_hybrid_config(db: Session) -> Dict[str, Any]:
        """Get hybrid retriever configuration."""
        return {
            "mode": ConfigService.get_config(db, "hybrid.mode"),
            "num_queries": ConfigService.get_config(db, "hybrid.num_queries"),
        }

    @staticmethod
    def get_prompt_config(db: Session) -> Dict[str, str]:
        """Get prompt configuration."""
        return {
            "base": ConfigService.get_config(db, "prompt.base"),
            "email": ConfigService.get_config(db, "prompt.email"),
            "whatsapp": ConfigService.get_config(db, "prompt.whatsapp"),
            "query_enhancement": ConfigService.get_config(
                db, "prompt.query_enhancement"
            ),
            "intent_detection": ConfigService.get_config(db, "prompt.intent_detection"),
        }

    @staticmethod
    def get_ingestion_config(db: Session) -> Dict[str, Any]:
        """Get ingestion-specific configuration."""
        return {
            "chunk_size": ConfigService.get_config(db, "ingestion.chunk_size", 1024),
            "chunk_overlap": ConfigService.get_config(
                db, "ingestion.chunk_overlap", 200
            ),
            "markdown_parser": ConfigService.get_config(
                db, "ingestion.markdown_parser", "markdown"
            ),
        }
