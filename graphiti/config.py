import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypedDict, cast
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

class MCPConfig(BaseModel):
    """Configuration for the Graphiti MCP server."""
    """Configuration for MCP server."""

    transport: str = 'sse'  # Default to SSE transport
    host: str = '0.0.0.0'
    port: int = 8000

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> 'MCPConfig':
        """Create MCP configuration from CLI arguments."""
        return cls(
                transport=args.transport,
                host=args.host,
                port=args.port
        )

DEFAULT_LLM_MODEL = "gemini-2.5-flash"
SMALL_LLM_MODEL = "gemini-2.5-flash-lite-preview-06-17"
DEFAULT_EMBEDDER_MODEL = "embedding-001"

class GraphitiEmbedderConfig(BaseModel):
    """Configuration for the embedder client.

    Centralizes all embedding-related configuration parameters.
    """

    model: str = DEFAULT_EMBEDDER_MODEL
    api_key: str | None = None
    max_tokens: int = 512

    @classmethod
    def from_env(cls) -> 'GraphitiEmbedderConfig':
        """Create embedder configuration from environment variables."""

        # Get model from environment, or use default if not set or empty
        model_env = os.environ.get('EMBEDDER_MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_EMBEDDER_MODEL

        return cls(
            model=model,
            api_key=os.environ.get('OPENAI_API_KEY'),
        )
    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'Neo4jConfig':
        """Create LLM configuration from CLI arguments, falling back to environment variables."""
        # Start with environment-based config
        config = cls.from_env()
        if hasattr(args, 'embedder_model') and args.embedder_model:
            config.model = args.embedder
        if hasattr(args, 'embedder_api_key') and args.embedder_api_key:
            config.api_key = args.embedder_api_key
        return config

class GraphitiLLMConfig(BaseModel):
    """Configuration for the LLM client.

    Centralizes all LLM-specific configuration parameters including API keys and model selection.
    """

    api_key: str | None = None
    model: str = DEFAULT_LLM_MODEL
    small_model: str = SMALL_LLM_MODEL
    temperature: float = 0.0
    max_tokens: int = 1000


    @classmethod
    def from_env(cls) -> 'GraphitiLLMConfig':
        """Create LLM configuration from environment variables."""
        # Get model from environment, or use default if not set or empty
        model_env = os.environ.get('MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_LLM_MODEL

        # Get small_model from environment, or use default if not set or empty
        small_model_env = os.environ.get('SMALL_MODEL_NAME', '')
        small_model = small_model_env if small_model_env.strip() else SMALL_LLM_MODEL

        # Setup for OpenAI API
        # Log if empty model was provided
        if model_env == '':
            logger.debug(
                f'MODEL_NAME environment variable not set, using default: {DEFAULT_LLM_MODEL}'
            )
        elif not model_env.strip():
            logger.warning(
                f'Empty MODEL_NAME environment variable, using default: {DEFAULT_LLM_MODEL}'
            )

        return cls(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model=model,
            small_model=small_model,
            temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiLLMConfig':
        """Create LLM configuration from CLI arguments, falling back to environment variables."""
        # Start with environment-based config
        config = cls.from_env()

        # CLI arguments override environment variables when provided
        if hasattr(args, 'model') and args.model:
            # Only use CLI model if it's not empty
            if args.model.strip():
                config.model = args.model
            else:
                # Log that empty model was provided and default is used
                logger.warning(f'Empty model name provided, using default: {DEFAULT_LLM_MODEL}')

        if hasattr(args, 'small_model') and args.small_model:
            if args.small_model.strip():
                config.small_model = args.small_model
            else:
                logger.warning(f'Empty small_model name provided, using default: {SMALL_LLM_MODEL}')

        if hasattr(args, 'temperature') and args.temperature is not None:
            config.temperature = args.temperature

        return config

class Neo4jConfig(BaseModel):
    """Configuration for Neo4j database connection."""

    uri: str = 'bolt://localhost:7687'
    user: str = 'neo4j'
    password: str = 'password'

    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Create Neo4j configuration from environment variables."""
        return cls(
            uri=os.environ.get('NEO4J_URI', 'bolt://neo4j:7687'),
            user=os.environ.get('NEO4J_USER', 'neo4j'),
            password=os.environ.get('NEO4J_PASSWORD', 'CHANGEME'),
        )
    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'Neo4jConfig':
        """Create LLM configuration from CLI arguments, falling back to environment variables."""
        # Start with environment-based config
        config = cls.from_env()
        if hasattr(args, 'neo4j_uri') and args.neo4j_uri:
            config.uri = args.neo4j_uri
        if hasattr(args, 'neo4j_user') and args.neo4j_user:
            config.user = args.neo4j_user
        if hasattr(args, 'neo4j_password') and args.neo4j_password:
            config.password = args.neo4j_password
        return config

class GraphitiConfig(BaseModel):
    llm: GraphitiLLMConfig = Field(default_factory=GraphitiLLMConfig)
    embedder: GraphitiEmbedderConfig = Field(default_factory=GraphitiEmbedderConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    group_id: str | None = None
    use_custom_entities: bool = False
    destroy_graph: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiConfig':
        """Create a configuration instance from environment variables."""
        return cls(
            llm=GraphitiLLMConfig.from_env(),
            embedder=GraphitiEmbedderConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiConfig':
        """Create configuration from CLI arguments, falling back to environment variables."""
        # Start with environment configuration
        config = cls.from_env()

        # Apply CLI overrides
        if args.group_id:
            config.group_id = args.group_id
        else:
            config.group_id = 'default'

        config.use_custom_entities = args.use_custom_entities
        config.destroy_graph = args.destroy_graph

        # Update LLM config using CLI args
        config.llm = GraphitiLLMConfig.from_cli_and_env(args)
        config.neo4j = Neo4jConfig.from_cli_and_env(args)
        config.embedder = GraphitiEmbedderConfig.from_cli_and_env(args)

        return config
