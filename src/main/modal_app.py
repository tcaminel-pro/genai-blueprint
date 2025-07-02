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
        # Install python-dotenv separately
        "cd /app && uv add python-dotenv",
    )
)


# Create a Modal stub
app = modal.App("genai-framework")


def get_secrets_dict():
    """Get secrets dictionary after dotenv is loaded."""
    return {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "AZURE_OPENAI_API_KEY": os.environ.get("AZURE_OPENAI_API_KEY", ""),
        "AZURE_OPENAI_ENDPOINT": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
        "LANGCHAIN_API_KEY": os.environ.get("LANGCHAIN_API_KEY", ""),
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", ""),
        # Add any other API keys from your .env file
    }


# Load dotenv first to get environment variables
from dotenv import load_dotenv

load_dotenv()

# Now create secrets with loaded environment variables
secrets = modal.Secret.from_dict(get_secrets_dict())


@app.function(
    image=image,
    secrets=[secrets],
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60,  # 1 hour timeout
    gpu="any",  # Optional: Use GPU if needed
    scaledown_window=300,  # Keep container alive for 5 minutes
)
@modal.concurrent(max_inputs=5)
@modal.web_server(8000)
def run():
    """Serve the Streamlit app via Modal web server."""
    import shlex
    import subprocess

    sys.path.append("/app")
    os.chdir("/app")

    # Set environment variables
    os.environ["BLUEPRINT_CONFIG"] = "container"
    os.environ["PYTHONPATH"] = "."

    # Create a .env file with the secrets
    secrets_dict = get_secrets_dict()
    with open(".env", "w") as f:
        for key, value in secrets_dict.items():
            if value:  # Only write non-empty values
                f.write(f"{key}={value}\n")

    # Create data directories if they don't exist
    os.makedirs(f"{VOLUME_PATH}/llm_cache", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/vector_store", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/hf_models", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/kv_store", exist_ok=True)

    # Start Streamlit server following Modal's recommended pattern
    target = shlex.quote("src/main/streamlit.py")
    cmd = f"uv run streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)


# @app.local_entrypoint()
# def main():
#     """Local entrypoint for Modal deployment."""
#     import time

#     print("Streamlit app deployed! Access it at:")
#     print("https://tcaminel--genai-framework-run-dev.modal.run")
#     print("Press Ctrl+C to stop the app")

#     try:
#         # Keep the app running indefinitely
#         while True:
#             time.sleep(60)
#     except KeyboardInterrupt:
#         print("\nStopping app...")
