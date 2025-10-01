# check=skip=SecretsUsedInArgOrEnv
# inspired by https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0 
# and (for uv):https://docs.astral.sh/uv/guides/integration/docker/

# use: docker build --pull --rm -f "Dockerfile" -t xxx:latest "."  --build-arg OPENAI_API=$OPENAI_API_KEY

FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder                                         
                                                                                                 
RUN apt-get update && apt-get install -y git curl build-essential libssl-dev \
    && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash \
    && export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")" \
    && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" \
    && nvm install --lts \
    && nvm alias default node \
    && nvm use default

# Install system dependencies and clean up
RUN apt-get update && apt-get install -y \
    graphviz-dev \
    ca-certificates \
    openssl \
    wget \
    && rm -rf /var/lib/apt/lists/*
                                                                                                 
# A directory to have app data                                                                   
WORKDIR /app                                                                                     
                                                                                                 
# Install dependencies first (for better caching)                                                
COPY pyproject.toml uv.lock ./                                                                   
RUN --mount=type=cache,target=/root/.cache/uv \                                                  
    uv sync --locked --no-install-project                                                        
                                                                                                 
# Copy the project and install it                                                                
COPY . .                                                                                         
RUN --mount=type=cache,target=/root/.cache/uv \                                                  
    uv sync --locked   

# The runtime image, used to just run the code provided its virtual environment                  
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim  AS runtime       

WORKDIR /app    
                                                                                                                               
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"                                                                  
                                                                                                 
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
# Copy Node.js installation from builder to a non-root location
COPY --from=builder /root/.nvm/versions/node /opt/node-versions
# Prepare Node.js for non-root user
RUN set -eux; \
    chmod -R a+rx /opt/node-versions; \
    ln -s /opt/node-versions/* /opt/node; \
    ln -sf /usr/bin/curl /usr/local/bin/curl; \
    ln -sf /usr/bin/wget /usr/local/bin/wget
                                                                                                 
# Create data directories with proper ownership (training data will be mounted at runtime)
# Create appuser first to set ownership during directory creation
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser
RUN mkdir -p /data/external \
    && mkdir -p /data/llm_cache \
    && mkdir -p /data/hf_models \
    && mkdir -p /data/vector_store \
    && mkdir -p /data/kv_store \
    && mkdir -p /tmp/mcp-workspace \
    && chown -R appuser:appuser /data /home/appuser /tmp/mcp-workspace

# Copy application files with proper ownership
COPY --from=builder --chown=appuser:appuser /app/use_case_data ./use_case_data                   
COPY --from=builder --chown=appuser:appuser /app/genai_blueprint ./genai_blueprint               
COPY --from=builder --chown=appuser:appuser /app/config ./config                                 
COPY --from=builder --chown=appuser:appuser /app/.streamlit ./.streamlit                         
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml ./                               
COPY --from=builder --chown=appuser:appuser /app/uv.lock ./
                                                                                                 
# Secrets will be mounted at runtime via Docker secrets                                          
ENV BASIC_AUTH_ENABLED=1 \
    BLUEPRINT_CONFIG="container" \
    PYTHONPATH="/app:/app/genai_blueprint" \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    NODE_PATH="/usr/local/lib/node_modules" \
    PATH="/opt/node/bin:/app/.venv/bin:/usr/local/bin:/usr/bin:/bin:/sbin"

USER appuser

EXPOSE 8501 

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8501/_stcore/health || exit 1

# Start Streamlit
CMD ["streamlit", "run", "/app/genai_blueprint/main/streamlit.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
