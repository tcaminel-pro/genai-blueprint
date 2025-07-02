"""
Modal deployment for GenAI Framework Streamlit application.
"""

import os
import sys

import modal

# Define the Modal volume to persist data
volume = modal.Volume.from_name("genai-data", create_if_missing=True)
VOLUME_PATH = "/data"

IGNORED_FILES = [
    ".aider*",
    "*.aider*",
    ".git",
    "__pycache__",
    "*.pyc",
    ".venv",
    ".env",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    "tests",
    "docs",
    "docker-compose.yml",
    ".blueprint.input.history",
    "*.tmp",
    "*.temp",
    # Exclude specific markdown files but keep README.md for package build
    "README_MODAL.md",
    "CONVENTIONS.md",
]

# Define the Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        # Install system dependencies
        "apt-get update && apt-get install -y curl make git",
        # Install uv
        "curl -fsSL https://github.com/astral-sh/uv/releases/download/0.1.24/uv-installer.sh | bash",
    )
    # Add source code - this layer will be rebuilt when source changes
    .add_local_dir(".", remote_path="/app", ignore=IGNORED_FILES, copy=True)
    .run_commands(
        # Install Python dependencies using uv after adding source
        "cd /app && uv sync",
        # Install FastAPI for web endpoints
        "cd /app && uv add 'fastapi[standard]'",
    )
)


# Create a Modal stub
app = modal.App("genai-framework")

# Define the Modal secrets - add all your API keys here
secrets_dict = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
    "AZURE_OPENAI_API_KEY": os.environ.get("AZURE_OPENAI_API_KEY", ""),
    "AZURE_OPENAI_ENDPOINT": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
    "LANGCHAIN_API_KEY": os.environ.get("LANGCHAIN_API_KEY", ""),
    "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", ""),
    # Add any other API keys from your .env file
}

secrets = modal.Secret.from_dict(secrets_dict)


@app.function(
    image=image,
    secrets=[secrets],
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60,  # 1 hour timeout
    gpu="any",  # Optional: Use GPU if needed
    container_idle_timeout=300,  # Keep container alive for 5 minutes
)
@modal.fastapi_endpoint(method="GET")
def web_app():
    """Serve the Streamlit app via Modal web endpoint."""
    sys.path.append("/app")
    os.chdir("/app")

    # Set environment variable for Modal-specific config
    os.environ["BLUEPRINT_CONFIG"] = "modal"

    # Create a .env file with the secrets
    with open(".env", "w") as f:
        for key, value in secrets_dict.items():
            if value:  # Only write non-empty values
                f.write(f"{key}={value}\n")

    # Create data directories if they don't exist
    os.makedirs(f"{VOLUME_PATH}/llm_cache", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/vector_store", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/hf_models", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/kv_store", exist_ok=True)

    # Start Streamlit and return a redirect response
    import subprocess
    import time
    from threading import Thread

    # Start Streamlit server in background
    def start_streamlit():
        subprocess.run(
            [
                "uv",
                "run",
                "streamlit",
                "run",
                "src/main/streamlit.py",
                "--server.port=8501",
                "--server.address=0.0.0.0",
                "--server.headless=true",
                "--server.enableCORS=false",
            ]
        )

    streamlit_thread = Thread(target=start_streamlit)
    streamlit_thread.daemon = True
    streamlit_thread.start()

    # Wait for Streamlit to start
    time.sleep(10)

    # Return HTML that redirects to the Streamlit app
    return """
    <html>
        <head>
            <title>GenAI Framework</title>
            <meta http-equiv="refresh" content="0; url=http://localhost:8501">
        </head>
        <body>
            <h1>GenAI Framework</h1>
            <p>Redirecting to Streamlit app...</p>
            <p>If not redirected automatically, <a href="http://localhost:8501">click here</a></p>
        </body>
    </html>
    """


@app.local_entrypoint()
def main():
    """Local entrypoint for Modal deployment."""
    run_app.remote()
