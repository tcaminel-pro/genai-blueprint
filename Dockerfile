# inspired by https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0 
# and (for uv): https://blog.west-webworld.fr/python-de-poetry-a-uv-il-ny-a-quun-pas/ 

# use: docker build --pull --rm -f "Dockerfile" -t xxx:latest "."  --build-arg OPENAI_API=$OPENAI_API_KEY


FROM python:3.12-bookworm  as builder

RUN apt-get update && apt-get install -y git curl

# Install system dependencies and clean up
RUN apt-get install -y graphviz-dev && \
    rm -rf /var/lib/apt/lists/*

# A directory to have app data 
WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-cache --no-group dev

# The runtime image, used to just run the code provided its virtual environment
FROM python:3.12-slim-bookworm  as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Secrets will be mounted at runtime via Docker secrets

ENV BASIC_AUTHENTICATION=1 \
    BLUEPRINT_CONFIG="container" \
    PYTHONPATH=".:/app"

COPY use_case_data ./use_case_data
COPY src ./src
COPY static ./static 
COPY app_conf.yaml ./app_conf.yaml  
COPY .streamlit ./.streamlit
COPY pyproject.toml uv.lock ./


EXPOSE 8501

#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV PYTHONPATH="${PYTHONPATH}:/src"

ENTRYPOINT ["streamlit", "run", "src/main/streamlit.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.enableCORS=false", \
            "--server.enableXsrfProtection=false"]
