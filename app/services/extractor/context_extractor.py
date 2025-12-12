import asyncio
from typing import List
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from pydantic import ConfigDict

from app.services.config_service import ConfigService
from app.database import get_db
# 1. The Anthropic Prompt
db  = next(get_db())
CONTEXT_PROMPT_TMPL = ConfigService.get_prompt_config(db)["context_extraction"]

class ContextExtractor(TransformComponent):
    """
    Custom generic transform that adds context to chunks based on the whole document.
    """
    
    model_config = ConfigDict(extra="allow")


    def __init__(
        self,
        llm: OpenAI,
        whole_document: str,
        concurrency_limit: int = 10,
    ):
        super().__init__()
        self.llm = llm
        self.whole_document = whole_document
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.prompt = PromptTemplate(CONTEXT_PROMPT_TMPL)

    async def _process_node(self, node: BaseNode):
        """Generates context for a single node and prepends it."""
        async with self.semaphore:
            try:
                # Generate context using the LLM
                context = await self.llm.apredict(
                    self.prompt,
                    whole_document=self.whole_document,
                    chunk_content=node.get_content(),
                )

                # Store original content in metadata (for reference/generation later)
                node.metadata["original_content"] = node.get_content()

                # Prepend context to the node content (this is what gets embedded)
                # Format: [Context] \n\n [Original Content]
                new_content = f"{context}---{node.get_content()}"
                node.set_content(new_content)

            except Exception as e:
                # Log error but don't fail the whole pipeline; keep original text
                print(f"Failed to generate context for node: {e}")
                pass
        return node

    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        """Synchronous entry point - unused in async pipeline but required by interface."""
        return asyncio.run(self.acall(nodes, **kwargs))

    async def acall(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        """Asynchronous entry point for the pipeline."""
        # Process all nodes in parallel (controlled by semaphore)
        tasks = [self._process_node(node) for node in nodes]
        processed_nodes = await asyncio.gather(*tasks)
        return processed_nodes
