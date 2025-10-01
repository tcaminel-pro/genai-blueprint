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
COPY --from=builder /root/.nvm /root/.nvm   

# Create symlinks for node, npm, and npx to make them available globally                         
RUN ln -sf /root/.nvm/versions/node/$(ls /root/.nvm/versions/node | head -1)/bin/node /usr/local/bin/node && \                                                                         
    ln -sf /root/.nvm/versions/node/$(ls /root/.nvm/versions/node | head -1)/bin/npm /usr/local/bin/npm && \                                                                          
    ln -sf /root/.nvm/versions/node/$(ls /root/.nvm/versions/node | head -1)/bin/npx /usr/local/bin/npx && \
    ln -sf /usr/bin/curl /usr/local/bin/curl && \
    ln -sf /usr/bin/wget /usr/local/bin/wget
                                                                                                 
# Create data directory (training data will be mounted at runtime)
RUN mkdir -p /data/external

# Copy application files                                                                         
COPY --from=builder /app/use_case_data ./use_case_data                                           
COPY --from=builder /app/src ./src                                                               
COPY --from=builder /app/config ./config                                                         
COPY --from=builder /app/.streamlit ./.streamlit                                                 
COPY --from=builder /app/pyproject.toml ./
                                                                                                 
# Secrets will be mounted at runtime via Docker secrets                                          
ENV BASIC_AUTH_ENABLED=1 \
    BLUEPRINT_CONFIG="container" \
    PYTHONPATH="/app:/app/src" \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501 

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8501/_stcore/health || exit 1

# Start Streamlit
CMD ["streamlit", "run", "/app/genai_blueprint/main/streamlit.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
