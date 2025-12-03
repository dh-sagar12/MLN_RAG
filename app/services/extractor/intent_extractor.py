
from typing import Any, Dict, List, Optional, Sequence
from llama_index.core.extractors import BaseExtractor
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.async_utils import run_jobs, DEFAULT_NUM_WORKERS
from llama_index.core.settings import Settings
from llama_index.core.bridge.pydantic import Field, SerializeAsAny

DEFAULT_INTENT_EXTRACT_TEMPLATE = """\
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

Respond with comma-separated intent category names (e.g., "new_information_request,clarification" or just "comparison" if only one applies): """


class IntentExtractor(BaseExtractor):
    """
    Intent extractor. Node-level extractor that extracts user intent or query intent
    from document nodes. Extracts `intent_categories` metadata field as a list of intents.
    A single query can have multiple intents.

    Args:
        llm (Optional[LLM]): LLM to use for intent extraction
        prompt_template (str): Template for intent extraction prompt
    """

    llm: SerializeAsAny[LLM] = Field(description="The LLM to use for generation.")
    prompt_template: str = Field(
        default=DEFAULT_INTENT_EXTRACT_TEMPLATE,
        description="Prompt template to use when extracting intent.",
    )

    def __init__(
        self,
        llm: Optional[LLM] = None,
        prompt: str = "",
        prompt_template: str = DEFAULT_INTENT_EXTRACT_TEMPLATE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        **kwargs: Any,
    ) -> None:
        """Initialize IntentExtractor."""
        # Use provided prompt if given, otherwise use prompt_template
        template = prompt if prompt else prompt_template
        
        super().__init__(
            llm=llm or Settings.llm,
            prompt_template=template,
            num_workers=num_workers,
            **kwargs,
        )
        if not self.llm:
            raise ValueError("LLM must be provided for IntentExtractor")

    @classmethod
    def class_name(cls) -> str:
        return "IntentExtractor"

    async def _aextract_intent_from_node(self, node: BaseNode) -> Dict[str, Any]:
        """Extract intents from a node and return its metadata dict."""
        if self.is_text_node_only and not isinstance(node, TextNode):
            return {}

        context_str = node.get_content(metadata_mode=self.metadata_mode)
        
        try:
            response = await self.llm.apredict(
                PromptTemplate(template=self.prompt_template),
                context_str=context_str,
            )
            
            # Extract all intent categories from response
            response_clean = response.strip()
            response_lower = response_clean.lower()
            
            # Valid intent categories
            valid_categories = [
                "new_information_request",
                "clarification",
                "follow_up",
                "comparison",
                "objection_concern",
                "booking_action",
                "general_inquiry"
            ]
            
            # Extract intents - can be comma-separated or on separate lines
            found_intents = []
            
            # Split by comma first
            parts = [part.strip() for part in response_clean.split(",")]
            
            # Check each part for valid categories
            for part in parts:
                part_lower = part.lower()
                # Try exact match first
                for category in valid_categories:
                    if category == part_lower or category in part_lower:
                        if category not in found_intents:
                            found_intents.append(category)
                        break
                else:
                    # Try partial match if no exact match
                    for category in valid_categories:
                        category_words = category.replace("_", " ").split()
                        if any(word in part_lower for word in category_words if len(word) > 3):
                            if category not in found_intents:
                                found_intents.append(category)
                            break
            
            # If no intents found, try searching the whole response
            if not found_intents:
                for category in valid_categories:
                    if category in response_lower:
                        found_intents.append(category)
            
            # If still no intents found, default to general_inquiry
            if not found_intents:
                found_intents = ["general_inquiry"]
            
            return {
                "intent_categories": found_intents,
            }
        except Exception as e:
            # Return default value on error
            return {
                "intent_categories": ["general_inquiry"],
            }

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Extract intent metadata from a sequence of nodes."""
        print("IntentExtractor extracting intents from nodes: ", len(nodes))
        intent_jobs = []
        for node in nodes:
            intent_jobs.append(self._aextract_intent_from_node(node))

        metadata_list: List[Dict] = await run_jobs(
            intent_jobs, 
            show_progress=self.show_progress, 
            workers=self.num_workers
        )

        return metadata_list