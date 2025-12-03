


from llama_index.core.vector_stores.types import MetadataFilter
from enum import Enum



class ListIntersectsFilterOperator(str, Enum):
    """
    List intersects filter operator.
    """
    LIST_INTERSECTS = "list_intersects"



class IntentMetadataFilter(MetadataFilter):
    """
    Intent metadata filter.
    """
    key: str
    value: list[str]
    operator: ListIntersectsFilterOperator = ListIntersectsFilterOperator.LIST_INTERSECTS