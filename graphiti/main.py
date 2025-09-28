import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypedDict, cast

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from .config import MCPConfig, GraphitiConfig
from .model import NodeResult, NodeSearchResponse, FactSearchResponse, EpisodeSearchResponse, StatusResponse

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client.gemini_client import GeminiClient,LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder,GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder,OpenAIEmbedderConfig
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.nodes import EpisodeType, EpisodicNode
#from fastmcp.server.auth.providers.keycloak import KeycloakAuthProvider
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.server.auth import OAuthProxy
#from fastmcp.server.auth import AuthSettings

#import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

load_dotenv()

GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph. Graphiti performs well
with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti transforms information into a richly connected knowledge network, allowing you to
capture relationships between concepts, entities, and information. The system organizes data as episodes
(content snippets), nodes (entities), and facts (relationships between entities), creating a dynamic,
queryable memory store that evolves with new information. Graphiti supports multiple data formats, including
structured JSON data, enabling seamless integration with existing data pipelines and systems.

Facts contain temporal metadata, allowing you to track the time of creation and whether a fact is invalid
(superseded by new information).

Key capabilities:
1. Add episodes (text, messages, or JSON) to the knowledge graph with the add_memory tool
2. Search for nodes (entities) in the graph using natural language queries with search_nodes
3. Find relevant facts (relationships between entities) with search_facts
4. Retrieve specific entity edges or episodes by UUID
5. Manage the knowledge graph with tools like delete_episode, delete_entity_edge, and clear_graph

The server connects to a database for persistent storage and uses language models for certain operations.
Each piece of information is organized by group_id, allowing you to maintain separate knowledge domains.

When adding information, provide descriptive names and detailed content to improve search quality.
When searching, use specific queries and consider filtering by group_id for more relevant results.

For optimal performance, ensure the database is properly configured and accessible, and valid
API keys are provided for any language model operations.
"""

token_verifier = JWTVerifier(
    jwks_uri="https://oauth2-api-qa.comsatelsmart.com/auth/realms/microservicios/protocol/openid-connect/certs",
    issuer="https://oauth2-api-qa.comsatelsmart.com",
    audience="mcpneo4j"
)

auth = OAuthProxy(
    upstream_authorization_endpoint="https://oauth2-api-qa.comsatelsmart.com/oauth/authorize",
    upstream_token_endpoint="https://provider.com/oauth/token",
    upstream_client_id="mcpneo4j",
    upstream_client_secret="8164d629-ad59-4fec-bcf2-47a58cda2c4c",
    token_verifier=token_verifier,
    base_url="http://mcpneo4j.pm.comsatel.com.pe",
)

mcp = FastMCP(
    'Graphiti Agent Memory',
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
    host="0.0.0.0",
    port=8000,
    #auth_server_provider = auth,
    #auth={
    #    "auth_server_provider": auth,
    #    #"token_verifier": token_verifier,
    #    "issuer_url": token_verifier.issuer,
    #    "resource_server_url": auth.base_url,
    #}
    #   #auth=AuthSettings(provider=auth),
)

# Configuración con tu Keycloak
# Reemplaza con tus valores
#auth = KeycloakAuthProvider(
#    realm_url=os.getenv("KEYCLOAK_REALM_URL", "https://oauth2-api-qa.comsatelsmart.com/realms/microservicios"),
#    # La base_url es la URL de tu servidor FastMCP.
#    base_url=os.getenv("FASTMCP_BASE_URL", "http://mcpneo4j.pm.comsatel.com.pe/"),
#    required_scopes=["openid", "profile", "mcpneo4j"] # Incluye el scope de acceso
#)

#mcp.configure_auth(auth)


# graphiti_client: Graphiti | None = None

API_KEY = None # "CHANGEME_API_KEY"

#os.environ['OPENAI_API_KEY'] = API_KEY
os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false'

episode_queues: dict[str, asyncio.Queue] = {}
queue_workers: dict[str, bool] = {}

config = GraphitiConfig()
graphiti_client: Graphiti | None = None

class ErrorResponse(TypedDict):
    error: str

class SuccessResponse(TypedDict):
    message: str

ENTITY_TYPES : dict[str, BaseModel] = None
#ENTITY_TYPES: dict[str, BaseModel] = {
#    'Requirement': Requirement,  # type: ignore
#    'Preference': Preference,  # type: ignore
#    'Procedure': Procedure,  # type: ignore
#}

async def process_episode_queue(group_id: str):
    """Process episodes for a specific group_id sequentially.

    This function runs as a long-lived task that processes episodes
    from the queue one at a time.
    """
    global queue_workers

    logger.info(f'Starting episode queue worker for group_id: {group_id}')
    queue_workers[group_id] = True

    try:
        while True:
            # Get the next episode processing function from the queue
            # This will wait if the queue is empty
            process_func = await episode_queues[group_id].get()

            try:
                # Process the episode
                await process_func()
            except Exception as e:
                logger.error(f'Error processing queued episode for group_id {group_id}: {str(e)}')
            finally:
                # Mark the task as done regardless of success/failure
                episode_queues[group_id].task_done()
    except asyncio.CancelledError:
        logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
    except Exception as e:
        logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
    finally:
        queue_workers[group_id] = False
        logger.info(f'Stopped episode queue worker for group_id: {group_id}')

def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into a readable result.

    Since EntityEdge is a Pydantic BaseModel, we can use its built-in serialization capabilities.

    Args:
        edge: The EntityEdge to format

    Returns:
        A dictionary representation of the edge with serialized dates and excluded embeddings
    """
    result = edge.model_dump(
        mode='json',
        exclude={
            'fact_embedding',
        },
    )
    result.get('attributes', {}).pop('fact_embedding', None)
    return result


@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = 'text',
    source_description: str = '',
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add an episode to memory. This is the primary way to add information to the graph.

    This function returns immediately and processes the episode addition in the background.
    Episodes for the same group_id are processed sequentially to avoid race conditions.

    Args:
        name (str): Name of the episode
        episode_body (str): The content of the episode to persist to memory. When source='json', this must be a
                           properly escaped JSON string, not a raw Python dictionary. The JSON data will be
                           automatically processed to extract entities and relationships.
        group_id (str, optional): A unique ID for this graph. If not provided, uses the default group_id from CLI
                                 or a generated one.
        source (str, optional): Source type, must be one of:
                               - 'text': For plain text content (default)
                               - 'json': For structured data
                               - 'message': For conversation-style content
        source_description (str, optional): Description of the source
        uuid (str, optional): Optional UUID for the episode

   Notes:
        When using source='json':
        - The JSON must be a properly escaped string, not a raw Python dictionary
        - The JSON will be automatically processed to extract entities and relationships
        - Complex nested structures are supported (arrays, nested objects, mixed data types), but keep nesting to a minimum
        - Entities will be created from appropriate JSON properties
        - Relationships between entities will be established based on the JSON structure
    """
    global graphiti_client, episode_queues, queue_workers
    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        source_type = EpisodeType.text
        if source.lower() == 'message':
            source_type = EpisodeType.message
        elif source.lower() == 'json':
            source_type = EpisodeType.json

        source_type = EpisodeType.message if source.lower() == 'message' else (EpisodeType.json if source.lower() == 'json' else EpisodeType.text)
        effective_group_id = group_id if group_id is not None else config.group_id
        group_id_str = str(effective_group_id) if effective_group_id is not None else ''
        assert graphiti_client is not None, 'graphiti_client should not be None here'
        client = cast(Graphiti, graphiti_client)

        async def process_episode():
            try:
                logger.info(f"Processing queued episode '{name}' for group_id: {group_id_str}")
                entity_types = ENTITY_TYPES if config.use_custom_entities else {}
                await client.add_episode(
                    name=name,
                    episode_body=episode_body,
                    source=source_type,
                    source_description=source_description,
                    group_id=group_id_str,  # Using the string version of group_id
                    uuid=uuid,
                    reference_time=datetime.now(timezone.utc),
                    entity_types=entity_types,
                )
                logger.info(f"Episode '{name}' added successfully")

                logger.info(f"Episode '{name}' processed successfully")
            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error processing episode '{name}' for group_id {group_id_str}: {error_msg}"
                )

        # Initialize queue for this group_id if it doesn't exist
        if group_id_str not in episode_queues:
            episode_queues[group_id_str] = asyncio.Queue()
        # Add the episode processing function to the queue
        await episode_queues[group_id_str].put(process_episode)
        # Start a worker for this queue if one isn't already running
        if not queue_workers.get(group_id_str, False):
            asyncio.create_task(process_episode_queue(group_id_str))

        # Return immediately with a success message
        return SuccessResponse(
            message=f"Episode '{name}' queued for processing (position: {episode_queues[group_id_str].qsize()})"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error queuing episode task: {error_msg}')
        return ErrorResponse(error=f'Error queuing episode task: {error_msg}')

@mcp.tool()
async def search_memory_nodes(
    query: str,
    group_ids: list[str] | None = None,
    max_nodes: int = 10,
    center_node_uuid: str | None = None,
    entity: str = '',  # cursor seems to break with None
) -> NodeSearchResponse | ErrorResponse:
    """Search the graph memory for relevant node summaries.
    These contain a summary of all of a node's relationships with other nodes.

    Note: entity is a single entity type to filter results (permitted: "Preference", "Procedure").

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_nodes: Maximum number of nodes to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
        entity: Optional single entity type to filter results (permitted: "Preference", "Procedure")
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')
    try:
        effective_group_ids = (
            group_ids if group_ids is not None else [config.group_id] if config.group_id else []
        )

        if center_node_uuid is not None:
            search_config = NODE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
        else:
            search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        search_config.limit = max_nodes

        filters = SearchFilters()
        if entity != '':
            filters.node_labels = [entity]

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Perform the search using the _search method
        search_results = await client._search(
            query=query,
            config=search_config,
            group_ids=effective_group_ids,
            center_node_uuid=center_node_uuid,
            search_filter=filters,
        )

        if not search_results.nodes:
            return NodeSearchResponse(message='No relevant nodes found', nodes=[])

        # Format the node results
        formatted_nodes: list[NodeResult] = [
            {
                'uuid': node.uuid,
                'name': node.name,
                'summary': node.summary if hasattr(node, 'summary') else '',
                'labels': node.labels if hasattr(node, 'labels') else [],
                'group_id': node.group_id,
                'created_at': node.created_at.isoformat(),
                'attributes': node.attributes if hasattr(node, 'attributes') else {},
            }
            for node in search_results.nodes
        ]

        return NodeSearchResponse(message='Nodes retrieved successfully', nodes=formatted_nodes)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching nodes: {error_msg}')
        return ErrorResponse(error=f'Error searching nodes: {error_msg}')

@mcp.tool()
async def search_memory_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Search the graph memory for relevant facts.

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_facts: Maximum number of facts to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Validate max_facts parameter
        if max_facts <= 0:
            return ErrorResponse(error='max_facts must be a positive integer')

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids if group_ids is not None else [config.group_id] if config.group_id else []
        )

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        relevant_edges = await client.search(
            group_ids=effective_group_ids,
            query=query,
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
        )

        if not relevant_edges:
            return FactSearchResponse(message='No relevant facts found', facts=[])

        facts = [format_fact_result(edge) for edge in relevant_edges]
        return FactSearchResponse(message='Facts retrieved successfully', facts=facts)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching facts: {error_msg}')
        return ErrorResponse(error=f'Error searching facts: {error_msg}')

@mcp.tool()
async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an entity edge from the graph memory.

    Args:
        uuid: UUID of the entity edge to delete
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(client.driver)
        return SuccessResponse(message=f'Entity edge with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error deleting entity edge: {error_msg}')

@mcp.tool()
async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an episode from the graph memory.

    Args:
        uuid: UUID of the episode to delete
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the episodic node by UUID - EpisodicNode is already imported at the top
        episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(client.driver)
        return SuccessResponse(message=f'Episode with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting episode: {error_msg}')
        return ErrorResponse(error=f'Error deleting episode: {error_msg}')

@mcp.tool()
async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    """Get an entity edge from the graph memory by its UUID.

    Args:
        uuid: UUID of the entity edge to retrieve
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge directly using the EntityEdge class method
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)

        # Use the format_fact_result function to serialize the edge
        # Return the Python dict directly - MCP will handle serialization
        return format_fact_result(entity_edge)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error getting entity edge: {error_msg}')

@mcp.tool()
async def get_episodes(
    group_id: str | None = None, last_n: int = 10
) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
    """Get the most recent memory episodes for a specific group.

    Args:
        group_id: ID of the group to retrieve episodes from. If not provided, uses the default group_id.
        last_n: Number of most recent episodes to retrieve (default: 10)
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id if group_id is not None else config.group_id

        if not isinstance(effective_group_id, str):
            return ErrorResponse(error='Group ID must be a string')

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        episodes = await client.retrieve_episodes(
            group_ids=[effective_group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
        )

        if not episodes:
            return EpisodeSearchResponse(
                message=f'No episodes found for group {effective_group_id}', episodes=[]
            )

        # Use Pydantic's model_dump method for EpisodicNode serialization
        formatted_episodes = [
            # Use mode='json' to handle datetime serialization
            episode.model_dump(mode='json')
            for episode in episodes
        ]

        # Return the Python list directly - MCP will handle serialization
        return formatted_episodes
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting episodes: {error_msg}')
        return ErrorResponse(error=f'Error getting episodes: {error_msg}')

@mcp.tool()
async def clear_graph() -> SuccessResponse | ErrorResponse:
    """Clear all data from the graph memory and rebuild indices."""
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # clear_data is already imported at the top
        await clear_data(client.driver)
        await client.build_indices_and_constraints()
        return SuccessResponse(message='Graph cleared successfully and indices rebuilt')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error clearing graph: {error_msg}')
        return ErrorResponse(error=f'Error clearing graph: {error_msg}')

@mcp.resource('http://graphiti/status')
async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and Neo4j connection."""
    global graphiti_client

    if graphiti_client is None:
        return StatusResponse(status='error', message='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Test database connection
        await client.driver.client.verify_connectivity()  # type: ignore

        return StatusResponse(
            status='ok', message='Graphiti MCP server is running and connected to Neo4j'
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error checking Neo4j connection: {error_msg}')
        return StatusResponse(
            status='error',
            message=f'Graphiti MCP server is running but Neo4j connection failed: {error_msg}',
        )

async def initialize_graphiti(config: GraphitiConfig):
    global graphiti_client
    llm_config = LLMConfig(
        model=config.llm.model,
        api_key=config.llm.api_key,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )

    llm_client = GeminiClient(config=llm_config)

    embedder = GeminiEmbedder(
        config=GeminiEmbedderConfig(
            embedding_model=config.embedder.model,
            api_key=config.embedder.api_key,
            max_tokens=config.embedder.max_tokens,
        )
    )

    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            embedding_model="embeddinggemma",
            api_key="none",
            max_tokens=512,
            base_url="http://ollama.pm.comsatel.com.pe/v1"
        )
    )

    cross_encoder=GeminiRerankerClient(
        config=LLMConfig(
            api_key=config.llm.api_key,
            model="gemini-2.5-flash-lite-preview-06-17",
        )
    )

    graphiti_client = Graphiti(
        uri = config.neo4j.uri,
        user=config.neo4j.user,
        password=config.neo4j.password,
        llm_client=llm_client,
        embedder=embedder, #gemini_embedder,
        cross_encoder=cross_encoder,
    )

    await graphiti_client.build_indices_and_constraints()
    logger.info('Graphiti client initialized successfully')

async def initialize_server() -> MCPConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    global config,mcp

    parser = argparse.ArgumentParser(description='Graphiti MCP Server')
    parser.add_argument('--host',type=str,help='uvicorn host IP',default='0.0.0.0')
    parser.add_argument('--port',type=int,help='uvicorn port',default=8000)
    parser.add_argument('--group_id',type=str,help='The default group ID to use for Graphiti operations.',)
    parser.add_argument('--transport',type=str,choices=["sse","stdio"],help="Transport use to communicate. Default(sse)",default="sse")
    parser.add_argument('--use_custom_entities',type=bool,help="Whether to enable custom entity extraction using LLMs.",default=False)
    parser.add_argument("--destroy_graph",type=bool,default=False,help="Wheter destroy graph or not. Default(false)")
    parser.add_argument("--neo4j_uri",type=str,help="Neo4j URI",default=None) #"neo4j://neo4j.pm.comsatel.com.pe:7687")
    parser.add_argument("--neo4j_user",type=str,help="Neo4j User",default="neo4j")
    parser.add_argument("--neo4j_password",type=str,help="Neo4j Password", default="welcome1")
    parser.add_argument("--api_key",type=str,help="LLM API Key")

    args = parser.parse_args()
    config = GraphitiConfig.from_cli_and_env(args)

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    if args.group_id:
        logger.info(f'Using provided group_id: {config.group_id}')
    else:
        logger.info(f'Generated random group_id: {config.group_id}')

    return MCPConfig.from_cli(args)

async def run_mcp_server():
    """Run the MCP server with the initialized configuration."""
    global config

    #if config is None:
    mcp_config = await initialize_server()
    logger.info(config)
    await initialize_graphiti(config)

    logger.info(f'CLI Args parsed for Uvicorn: host={mcp_config.host}, port={mcp_config.port}')
    logger.info(f'Starting MCP server with transport: {mcp_config.transport} on {mcp_config.host}:{mcp_config.port}')

    logger.info(f'Starting MCP server with transport: {mcp_config.transport}')
    if mcp_config.transport == 'stdio':
        await mcp.run_stdio_async()
        await mcp.run_stdio_async(host=mcp_config.host, port=mcp_config.port)
    elif mcp_config.transport == 'sse': # in ['sse', 'http']:
        logger.info(f"Starting MCP server with {mcp_config.transport} transport")
        #mcp.run(transport=mcp_config.transport,host=mcp_config.host,port=mcp_config.port)
        #uvicorn.run(
        #    mcp.app,
        #    host=mcp_config.host,
        #    port=mcp_config.port,
        #    log_level="info",
        #    # Nota: El transporte (SSE/HTTP) es manejado por el routing de FastMCP en mcp.app
        #)
        #await mcp.run_sse_async()
        await mcp.run_sse_async()
    elif mcp_config.transport == 'http':
        logger.info("Starting MCP server with HTTP transport")
        await mcp.run_http_async()

def main():
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except Exception as e:
        logger.error(f'Error initializing Graphiti MCP server: {str(e)}')
        raise

if __name__ == '__main__':
    main()
