# inspired by https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0 

# use: docker build --pull --rm -f "Dockerfile" -t xxx:latest "."  --build-arg OPENAI_API=$OPENAI_API_KEY


FROM python:3.11-bookworm  as builder

RUN apt-get update && apt-get install -y git

RUN apt-get install -y graphviz-dev

RUN pip install poetry==1.5.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# A directory to have app data 
WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --without dev  --without extra --no-root && rm -rf $POETRY_CACHE_DIR

# The runtime image, used to just run the code provided its virtual environment
FROM python:3.11-slim-bookworm  as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

###  TEMPORY KEY FOR TESTS !! TO BE REMOVED

ARG OPENAI_API
ARG AZURE_OPENAI_API
ARG STREAMLIT_ENTRY_POINT
ARG GROQ_API_KEY
ARG LANGCHAIN_API_KEY

ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API
ENV STREAMLIT_ENTRY_POINT=$STREAMLIT_ENTRY_POINT
ENV GROQ_API_KEY=$GROQ_API_KEY
ENV LANGCHAIN_API_KEY=$LANGCHAIN_API_KEY

ENV BASIC_AUTHENTICATION=1
ENV CONFIGURATION="cloud_openai"

COPY use_case_data ./use_case_data
COPY python ./python 
COPY static ./static 
COPY app_conf.yaml ./app_conf.yaml  
COPY .streamlit ./.streamlit   


EXPOSE 8501

#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV PYTHONPATH "${PYTHONPATH}:/python"

ENTRYPOINT ["streamlit", "run", "python/GenAI_Lab.py", "--server.port=8501", "--server.address=0.0.0.0"]