# Graphiti Agent Memory

Graphiti is a memory service for AI agents built on a knowledge graph. It excels at managing dynamic data from various sources, such as user interactions, enterprise data, and external information. By transforming information into a connected network of episodes, nodes, and facts, Graphiti creates a queryable memory store that evolves as new data is added.

## Features

- **Knowledge Graph Storage**: Organizes information as episodes, nodes (entities), and facts (relationships) in a knowledge graph.
- **Dynamic Data Handling**: Supports multiple data formats, including plain text, JSON, and conversational messages.
- **Temporal Metadata**: Tracks the creation time of facts and marks them as invalid if they are superseded by new information.
- **Asynchronous Processing**: Episodes are added to a queue and processed in the background to avoid blocking.
- **Search Capabilities**: Provides tools to search for nodes and facts using natural language queries.
- **Configuration Flexibility**: The server can be configured using environment variables or command-line arguments.

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd graphiti-agent-memory
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The server can be configured using environment variables or command-line arguments.

### Environment Variables

- `NEO4J_URI`: The URI for the Neo4j database (e.g., `bolt://localhost:7687`).
- `NEO4J_USER`: The username for the Neo4j database.
- `NEO4J_PASSWORD`: The password for the Neo4j database.
- `OPENAI_API_KEY`: The API key for the language model service.
- `MODEL_NAME`: The default language model to use.
- `SMALL_MODEL_NAME`: A smaller, secondary language model for less complex tasks.
- `EMBEDDER_MODEL_NAME`: The name of the embedding model to use.

### Command-Line Arguments

- `--host`: The host IP address for the server (default: `0.0.0.0`).
- `--port`: The port number for the server (default: `8000`).
- `--group_id`: The default group ID for graph operations.
- `--transport`: The communication transport to use (`sse` or `stdio`).
- `--use_custom_entities`: Enable or disable custom entity extraction.
- `--destroy_graph`: Clear the graph on startup.

## Usage

To run the server, use the following command:

```bash
python -m graphiti.main
```

You can also provide command-line arguments to override the default configuration:

```bash
python -m graphiti.main --port 8080 --group_id my-graph
```

## API Tools

The server provides the following tools for interacting with the knowledge graph:

- `add_memory`: Adds an episode to the knowledge graph.
- `search_memory_nodes`: Searches for nodes in the graph based on a query.
- `search_memory_facts`: Searches for facts (edges) in the graph based on a query.
- `delete_entity_edge`: Deletes an entity edge from the graph.
- `delete_episode`: Deletes an episode from the graph.
- `get_entity_edge`: Retrieves an entity edge by its UUID.
- `get_episodes`: Retrieves the most recent episodes from the graph.
- `clear_graph`: Clears all data from the graph.
- `get_status`: Retrieves the status of the server and its connection to the database.