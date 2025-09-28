"""
Paquete principal para el servidor Graphiti MCP basado en Neo4j.

Define la versión del paquete y expone cualquier clase o función clave
que deba ser importada directamente desde el nivel superior.
"""
# Importa la versión del paquete desde pyproject.toml
# Esta línea se usa a menudo para que las herramientas puedan acceder a la versión
# de forma programática.
from importlib import metadata

__version__ = metadata.version(__package__ or __name__)

# Opcional: Si quieres que tus módulos principales (como 'config' o 'model')
# sean accesibles directamente como 'from mcp_neo4j import config',
# puedes importarlos aquí:
# from . import config
# from . import model

# Para la mayoría de las aplicaciones de servidor, dejarlo simple como arriba es suficiente.

