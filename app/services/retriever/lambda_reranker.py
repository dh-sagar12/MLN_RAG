"""Lambda-based reranker postprocessor for LlamaIndex.

This module provides a custom postprocessor that calls an AWS Lambda function
for reranking retrieved nodes, offloading the compute-intensive reranking
from the VPC to a serverless Lambda function.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config as BotoConfig
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import  Field, PrivateAttr

logger = logging.getLogger(__name__)


class LambdaReranker(BaseNodePostprocessor):
    """Reranker that uses AWS Lambda for scoring documents.

    This postprocessor sends the query and retrieved nodes to a Lambda function
    that runs the reranking model (e.g., BAAI/bge-reranker-large) and returns
    relevance scores for each node.

    Attributes:
        function_name: Name or ARN of the Lambda function.
        region_name: AWS region where the Lambda is deployed.
        top_n: Number of top results to return after reranking.
        timeout: Lambda invocation timeout in seconds.
        aws_access_key_id: Optional AWS access key ID (uses default credentials if not provided).
        aws_secret_access_key: Optional AWS secret access key.
    """


    function_name: str = Field(
        description="Lambda function name or ARN",
    )
    region_name: str = Field(
        default="ap-south-1",
        description="AWS region",
    )
    top_n: int = Field(
        default=5,
        description="Number of top results to return",
    )
    timeout: int = Field(
        default=60,
        description="Lambda timeout in seconds",
    )
    aws_access_key_id: Optional[str] = Field(
        default=None,
        description="AWS access key ID",
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None,
        description="AWS secret access key",
    )
    aws_session_token: Optional[str] = Field(
        default=None,
        description="AWS session token",
    )
    
    _lambda_client: Any = PrivateAttr()
    

    def __init__(
        self,
        function_name: str,
        region_name: str = "us-east-1",
        top_n: int = 5,
        timeout: int = 60,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Lambda reranker.

        Args:
            function_name: Lambda function name or ARN.
            region_name: AWS region where the Lambda is deployed.
            top_n: Number of top results to return after reranking.
            timeout: Lambda invocation timeout in seconds.
            aws_access_key_id: Optional AWS access key ID.
            aws_secret_access_key: Optional AWS secret access key.
            aws_session_token: Optional AWS session token.
        """
        print('aws_access_key_id------------>', aws_access_key_id)
        print('aws_secret_access_key------------>', aws_secret_access_key)
        print('aws_session_token------------>', aws_session_token)
        super().__init__(
            function_name=function_name,
            region_name=region_name,
            top_n=top_n,
            timeout=timeout,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            **kwargs,
        )
        self._lambda_client = self._create_lambda_client()

    def _create_lambda_client(self) -> Any:
        """Create and return a boto3 Lambda client.

        Returns:
            Configured boto3 Lambda client.
        """
        boto_config = BotoConfig(
            read_timeout=self.timeout,
            connect_timeout=10,
            retries={"max_attempts": 2},
        )

        client_kwargs: Dict[str, Any] = {
            "service_name": "lambda",
            "region_name": self.region_name,
            "config": boto_config,
        }

        if self.aws_access_key_id and self.aws_secret_access_key and self.aws_session_token:
            client_kwargs["aws_access_key_id"] = self.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
            client_kwargs["aws_session_token"] = self.aws_session_token

        return boto3.client(**client_kwargs)

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        return "LambdaReranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Rerank nodes using the Lambda function.

        Args:
            nodes: List of nodes to rerank.
            query_bundle: Query bundle containing the query string.

        Returns:
            Reranked and filtered list of nodes.
        """
        if not nodes:
            return []

        if query_bundle is None:
            logger.warning(
                "No query bundle provided for reranking, returning original nodes"
            )
            return nodes[: self.top_n]

        query = query_bundle.query_str

        try:
            # Prepare payload for Lambda
            payload = self._build_lambda_payload(query=query, nodes=nodes)
            print("payload------------>", payload)

            # Invoke Lambda function
            response = self._lambda_client.invoke(
                FunctionName=self.function_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload),
            )

            # Parse response
            response_payload = json.loads(response["Payload"].read().decode("utf-8"))

            # Check for Lambda errors
            if "errorMessage" in response_payload:
                logger.error(f"Lambda error: {response_payload['errorMessage']}")
                return nodes[: self.top_n]

            # Extract scores and rerank
            return self._apply_reranking_scores(
                nodes=nodes,
                response_payload=response_payload,
            )

        except Exception as e:
            logger.error(f"Error invoking Lambda reranker: {e}", exc_info=True)
            # Fallback: return original nodes truncated to top_n
            return nodes[: self.top_n]

    def _build_lambda_payload(
        self,
        *,
        query: str,
        nodes: List[NodeWithScore],
    ) -> Dict[str, Any]:
        """Build the payload to send to the Lambda function.

        Args:
            query: The query string.
            nodes: List of nodes to rerank.

        Returns:
            Dictionary payload for Lambda invocation.
        """
        documents = []
        for i, node_with_score in enumerate(nodes):
            documents.append(
                {
                    "index": i,
                    "text": node_with_score.node.get_content(),
                    "original_score": node_with_score.score,
                }
            )

        return {
            "query": query,
            "documents": documents,
            "top_n": self.top_n,
        }

    def _apply_reranking_scores(
        self,
        *,
        nodes: List[NodeWithScore],
        response_payload: Dict[str, Any],
    ) -> List[NodeWithScore]:
        """Apply reranking scores from Lambda response to nodes.

        Args:
            nodes: Original list of nodes.
            response_payload: Lambda response containing scores.

        Returns:
            Reranked list of nodes.
        """
        # Expected response format:
        # {
        #     "scores": [
        #         {"index": 0, "score": 0.95},
        #         {"index": 2, "score": 0.87},
        #         ...
        #     ]
        # }
        scores_data = response_payload.get("scores", [])

        if not scores_data:
            logger.warning("No scores returned from Lambda, returning original nodes")
            return nodes[: self.top_n]

        # Create a mapping of index to new score
        score_map: Dict[int, float] = {
            item["index"]: item["score"] for item in scores_data
        }

        # Update node scores
        reranked_nodes: List[NodeWithScore] = []
        for i, node_with_score in enumerate(nodes):
            if i in score_map:
                # Create new NodeWithScore with updated score
                new_node = NodeWithScore(
                    node=node_with_score.node,
                    score=score_map[i],
                )
                reranked_nodes.append(new_node)

        # Sort by new score descending
        reranked_nodes.sort(key=lambda x: x.score or 0, reverse=True)

        # Return top_n results
        return reranked_nodes[: self.top_n]
