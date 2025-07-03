# inspired by https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0 
# and (for uv):https://docs.astral.sh/uv/guides/integration/docker/

# use: docker build --pull --rm -f "Dockerfile" -t xxx:latest "."  --build-arg OPENAI_API=$OPENAI_API_KEY



FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder                                         
                                                                                                 
RUN apt-get update && apt-get install -y git curl                                                
                                                                                                 
# Install system dependencies and clean up                                                       
RUN apt-get install -y graphviz-dev && \                                                         
    rm -rf /var/lib/apt/lists/*                                                                  
                                                                                                 
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
                                                                                                                               
ENV VIRTUAL_ENV=/app/.venv \                                                                     
    PATH="/app/.venv/bin:$PATH"                                                                  
                                                                                                 
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}                                                
                                                                                                 
# Copy application files                                                                         
COPY --from=builder /app/use_case_data ./use_case_data                                           
COPY --from=builder /app/src ./src                                                               
COPY --from=builder /app/config ./config                                                         
COPY --from=builder /app/.streamlit ./.streamlit                                                 
COPY --from=builder /app/pyproject.toml ./                                                       
                                                                                                 
# Secrets will be mounted at runtime via Docker secrets                                          
ENV BASIC_AUTHENTICATION=1 \                                                                     
    BLUEPRINT_CONFIG="container" \                                                               
    PYTHONPATH=".:/app"        



EXPOSE 8501

#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV PYTHONPATH="${PYTHONPATH}:/src"

ENTRYPOINT ["streamlit", "run", "src/main/streamlit.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.enableCORS=false", \
            "--server.enableXsrfProtection=false"]
