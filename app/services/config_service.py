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
        # RAG Parameters
        "rag.top_k": {
            "value": 7,
            "type": "int",
            "description": "Number of top chunks to retrieve from the knowledge base. Higher values provide more context but may include less relevant information.",
            "category": "rag"
        },
        "rag.similarity_threshold": {
            "value": 0.75,
            "type": "float",
            "description": "Minimum similarity score (0-1) for retrieved chunks. Higher values return more relevant but fewer results.",
            "category": "rag"
        },
        "rag.response_mode": {
            "value": "compact",
            "type": "string",
            "description": "Response synthesis mode: 'compact' (faster, combines chunks), 'refine' (iterative refinement), 'tree_summarize' (tree-based), 'simple_summarize' (single call), 'accumulate' (concatenate all), or 'generation' (ignore context).",
            "category": "rag"
        },
        
        # Retriever Parameters
        "retriever.vector.top_k": {
            "value": 50,
            "type": "int",
            "description": "Maximum number of chunks to retrieve from vector search before filtering by similarity threshold.",
            "category": "retriever"
        },
        "retriever.vector.ef_search": {
            "value": 256,
            "type": "int",
            "description": "HNSW index search parameter. Higher values improve recall but slow down search. Typical range: 64-512.",
            "category": "retriever"
        },
        "retriever.bm25.top_k": {
            "value": 50,
            "type": "int",
            "description": "Maximum number of chunks to retrieve from BM25 keyword search.",
            "category": "retriever"
        },
        "retriever.bm25.enabled": {
            "value": True,
            "type": "bool",
            "description": "Enable BM25 keyword-based retrieval. When enabled, combines with vector search for hybrid retrieval.",
            "category": "retriever"
        },
        "retriever.bm25.language": {
            "value": "english",
            "type": "string",
            "description": "Language for BM25 full-text search. Must match PostgreSQL text search configuration.",
            "category": "retriever"
        },
        
        # Hybrid Retriever Parameters
        "hybrid.mode": {
            "value": "reciprocal_rerank",
            "type": "string",
            "description": "Fusion mode for combining results: 'reciprocal_rerank' (Reciprocal Rank Fusion), 'relative_score' (relative scoring), 'dist_based_score' (distance-based), or 'simple' (simple reordering).",
            "category": "hybrid"
        },
        "hybrid.num_queries": {
            "value": 1,
            "type": "int",
            "description": "Number of query variations to generate for retrieval. Higher values improve recall but increase latency.",
            "category": "hybrid"
        },
        
        # Prompt Configuration
        "prompt.base": {
            "value": "You are responding on behalf of Mountain Lodges of Nepal part of Sherpa Hospitality Group, a premium Himalayan hospitality and travel company. Format your response using Markdown for better readability (use bullet points, bold text, and paragraphs where appropriate).",
            "type": "string",
            "description": "Base system prompt that applies to all responses. Contains company branding and general instructions.",
            "category": "prompt"
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
            "category": "prompt"
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
            "category": "prompt"
        },
        "prompt.query_enhancement": {
            "value": """You are a query optimization system for a RAG-based response drafting assistant serving a trekking company in Nepal.

## Your Task
Transform the user's current message into a precise, retrieval-optimized query that fetches ONLY the missing information needed to draft an accurate  response.

## Analysis Framework

### Step 1: Parse Conversation State
Examine the conversation history and identify:
- **Already Addressed**: Topics/questions already answered with specific details (itinerary days, pricing, difficulty ratings, permit requirements, etc.)
- **Partially Addressed**: Topics mentioned but lacking depth or specific details the user is now asking about
- **Pending**: New questions or topics not yet covered
- **User Intent Shift**: Has the user changed topic, expressed dissatisfaction, or requested clarification?

### Step 2: Identify Current User Intent
Determine what the user wants RIGHT NOW:
- New information request (specific trek details, availability, pricing)
- Clarification of previous answer (more detail on X, alternative for Y)
- Follow-up question (building on previous answer)
- Comparison request (Trek A vs Trek B)
- Objection/concern handling (too difficult, too expensive, wrong season)
- Booking/action intent (ready to book, wants to proceed)

### Step 3: Extract Missing Information Gaps
Identify EXACTLY what's missing from the knowledge base to answer the current query:
- Specific trek names, routes, or regions
- Date ranges, seasonal information, or availability
- Difficulty levels, fitness requirements, or technical details
- Cost breakdowns, inclusions/exclusions, or payment terms
- Logistics (permits, guides, accommodation, transportation)
- Customization options or alternatives

### Step 4: Query Construction Rules

**INCLUDE in enhanced query:**
- Specific trek names, locations, or route identifiers
- Timeframes (months, seasons, specific dates if mentioned)
- Quantifiable parameters (days, altitude, distance, price range)
- Comparison criteria (if user asks "vs" or "alternative")
- Technical requirements (permits, fitness level, equipment)
- New topics or details not yet provided

**EXCLUDE from enhanced query:**
- Any information already retrieved and presented
- Details the user acknowledged or didn't question
- Context already established in prior responses
- User's personal background unless it changes the query scope

**Special Cases:**
- If user says "tell me more about X" → Query: "detailed information about X [specific aspect they want]"
- If user expresses concern → Query: "[concern topic] + alternatives/solutions"
- If user asks "what about Y?" after discussing X → Query: "Y [specific aspect]" (don't re-query X)
- If user asks for comparison → Query: "[Trek A] vs [Trek B] [specific comparison dimension]"

## Output Format

Provide ONLY the enhanced query in the plain text format without any other text or formatting:

<retrieval-optimized query>

### Query Optimization Guidelines:
- Use 3-8 words maximum (except for complex comparisons)
- Lead with the most specific identifier (trek name, location, topic)
- Include only searchable, factual terms (not conversational filler)
- Use keywords that match knowledge base terminology (e.g., "Annapurna Circuit itinerary 12 days" not "tell me what happens each day on the Annapurna trek")
- For clarifications, focus on the sub-topic: "Everest Base Camp acclimatization schedule" not "more details about EBC"

## Examples

### Example 1
**Previous:** User asked about Everest Base Camp trek. System provided: 12-day itinerary, difficulty level (moderate-challenging), best seasons (March-May, September-November).

**Current User Message:** "What about the permits and costs?"

**Analysis:** Itinerary and difficulty already covered. User now needs permit information and pricing.

**ENHANCED_QUERY:** `Everest Base Camp permits costs pricing`

---

### Example 2
**Previous:** User asked about treks in Annapurna region. System provided: overview of Annapurna Circuit and Annapurna Base Camp trek options.

**Current User Message:** "I only have 7 days. Which one can I do?"

**Analysis:** User has time constraint (7 days). Need short trek options in Annapurna region.

**ENHANCED_QUERY:** `Annapurna region treks 7 days short itinerary`

---

### Example 3
**Previous:** User asked about Langtang Valley trek. System provided: 7-day itinerary, moderate difficulty, best seasons.

**Current User Message:** "Is it safe? I heard about the earthquake."

**Analysis:** User has safety concerns about Langtang post-earthquake. Need current safety status and infrastructure info.

**ENHANCED_QUERY:** `Langtang Valley current safety status infrastructure`

---

### Example 4
**Previous:** User asked about Manaslu Circuit. System provided: 14-day itinerary, challenging difficulty, permit requirements.

**Current User Message:** "That sounds too hard. What's an easier alternative with similar views?"

**Analysis:** User finds Manaslu too difficult. Needs easier trek alternatives with comparable mountain scenery.

**ENHANCED_QUERY:** `moderate difficulty treks similar views Manaslu alternative`

---

### Example 5
**Previous:** User asked about October availability for Annapurna Circuit. System confirmed October is peak season with good availability.

**Current User Message:** "Great! What's included in the package price?"

**Analysis:** Availability confirmed. Now user needs package inclusions/exclusions breakdown.

**ENHANCED_QUERY:** `Annapurna Circuit package inclusions exclusions`

## Quality Checklist
Before outputting, verify:
- Query targets NEW information only
- Query is 3-8 words (concise and specific)
- Query uses knowledge-base-friendly keywords
- Query matches the user's current intent
- No redundant information already provided""",
            "type": "string",
            "description": "Prompt for query enhancement/optimization. Used to transform user queries into retrieval-optimized queries.",
            "category": "prompt"
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
                    description=config_data["description"]
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
    def set_config(db: Session, key: str, value: Any, description: Optional[str] = None) -> Configuration:
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
                description=description or default_data.get("description")
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
                "updated_at": config.updated_at.isoformat() if config.updated_at else None
            }
        
        return result
    
    @staticmethod
    def get_rag_config(db: Session) -> Dict[str, Any]:
        """Get RAG-specific configuration."""
        return {
            "top_k": ConfigService.get_config(db, "rag.top_k"),
            "similarity_threshold": ConfigService.get_config(db, "rag.similarity_threshold"),
            "response_mode": ConfigService.get_config(db, "rag.response_mode"),
        }
    
    @staticmethod
    def get_retriever_config(db: Session) -> Dict[str, Any]:
        """Get retriever-specific configuration."""
        return {
            "vector": {
                "top_k": ConfigService.get_config(db, "retriever.vector.top_k"),
                "ef_search": ConfigService.get_config(db, "retriever.vector.ef_search"),
            },
            "bm25": {
                "enabled": ConfigService.get_config(db, "retriever.bm25.enabled"),
                "top_k": ConfigService.get_config(db, "retriever.bm25.top_k"),
                "language": ConfigService.get_config(db, "retriever.bm25.language"),
            }
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
            "query_enhancement": ConfigService.get_config(db, "prompt.query_enhancement"),
        }

