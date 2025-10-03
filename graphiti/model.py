from pydantic import BaseModel, Field
from typing import Any, TypedDict, cast


class NodeResult(TypedDict):
    """Represents a single node retrieved from the knowledge graph.

    Attributes:
        uuid: The unique identifier of the node.
        name: The name of the node.
        summary: A brief summary of the node's relationships and content.
        labels: A list of labels associated with the node (e.g., entity types).
        group_id: The identifier for the graph group to which the node belongs.
        created_at: The timestamp when the node was created.
        attributes: A dictionary of additional attributes associated with the node.
    """

    uuid: str
    name: str
    summary: str
    labels: list[str]
    group_id: str
    created_at: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    """Defines the structure of a response for a node search operation.

    Attributes:
        message: A descriptive message about the outcome of the search.
        nodes: A list of nodes that match the search query.
    """

    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    """Defines the structure of a response for a fact search operation.

    Attributes:
        message: A descriptive message about the outcome of the search.
        facts: A list of facts (edges) that match the search query.
    """

    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(TypedDict):
    """Defines the structure of a response for an episode search operation.

    Attributes:
        message: A descriptive message about the outcome of the search.
        episodes: A list of episodes that match the search criteria.
    """

    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(TypedDict):
    """Defines the structure of a response for a server status check.

    Attributes:
        status: The current status of the server (e.g., 'ok', 'error').
        message: A message providing additional details about the server's status.
    """

    status: str
    message: str
