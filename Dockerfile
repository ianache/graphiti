# ----------------------------------------------------------------------
# FASE 1: BUILD (Construcción e Instalación de Dependencias)
# Utiliza la imagen base completa de uv para garantizar que todas las
# herramientas de compilación estén disponibles.
# ----------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS builder

WORKDIR /app

# Copia los metadatos y el archivo de bloqueo (uv.lock) primero.
# Esto optimiza el cache de Docker: si estos archivos no cambian,
# no se necesita reinstalar las dependencias.
COPY pyproject.toml /app/
COPY README.md /app/
COPY uv.lock /app/
COPY graphiti /app/

# Copia el paquete de código fuente completo.
# Aseguramos que la carpeta "graphiti" esté dentro de "/app".
COPY graphiti /app/graphiti

# Instalación de dependencias y el paquete local:
# 'uv install .' instala el proyecto definido en pyproject.toml
# ('graphiti'), usando el .venv y respetando el uv.lock.
# '--no-editable' es necesario para una imagen de producción.
#RUN --mount=type=cache,target=/root/.cache/uv \
#    uv install . --no-editable
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --frozen --no-editable

RUN uv pip install .



# ----------------------------------------------------------------------
# FASE 2: RUNTIME (Imagen Final, Producción)
# Utiliza la versión 'slim' para minimizar el tamaño final de la imagen.
# ----------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Copia el entorno virtual (.venv) pre-instalado desde la fase 'builder'.
COPY --from=builder /app/.venv /app/.venv
# Copia el código fuente del paquete (graphiti) desde la fase 'builder'.
COPY --from=builder /app /app

# Configuración del entorno
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV TRANSPORT=sse
ENV PORT=8000
# EXPOSE 8000 # Opcional si necesitas documentar el puerto

# El ENTRYPOINT ejecuta el script 'graphiti' definido en [project.scripts]
# de tu pyproject.toml.
#ENTRYPOINT ["graphiti","--port","8000","--transport","sse"]

ENTRYPOINT ["/app/.venv/bin/graphiti","--host","0.0.0.0","--port","8000","--transport","sse"]
# ,"--port","8000","--transport","sse"]
