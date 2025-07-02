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
    "*.md",
    "docker-compose.yml",
    ".blueprint.input.history",
    "*.tmp",
    "*.temp",
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
)
def run_app():
    """Run the Streamlit app using Modal."""
    sys.path.append("/app")

    # Change to the app directory
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

    # Install dependencies if needed
    os.system("cd /app && uv sync")

    # Run the Streamlit app
    os.system("streamlit run src/main/streamlit.py --server.port=8080 --server.address=0.0.0.0")


@app.local_entrypoint()
def main():
    """Local entrypoint for Modal deployment."""
    run_app.remote()
