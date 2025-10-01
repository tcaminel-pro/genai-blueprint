"""
Modal deployment for GenAI Framework Streamlit application.

Still WIP
"""

import os
import sys

import modal

# Deployment mode configuration
# Options: "code", "dockerfile", "aws_image"
DEPLOYMENT_MODE = os.environ.get("MODAL_DEPLOYMENT_MODE", "code")
DOCKERFILE_PATH = os.environ.get("MODAL_DOCKERFILE_PATH", "deploy/Dockerfile")
AWS_IMAGE_URI = os.environ.get("MODAL_AWS_IMAGE_URI", "")

print(f"{DEPLOYMENT_MODE=}")
print(f"{AWS_IMAGE_URI=}")


# AWS_IMAGE_URI = "909658914353.dkr.ecr.eu-west-1.amazonaws.com"
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


def get_image() -> modal.Image:
    """Get the Modal image based on deployment mode."""
    if DEPLOYMENT_MODE == "dockerfile":
        if not os.path.exists(DOCKERFILE_PATH):
            raise FileNotFoundError(f"Dockerfile not found at {DOCKERFILE_PATH}")
        return modal.Image.from_dockerfile(DOCKERFILE_PATH)

    elif DEPLOYMENT_MODE == "aws_image":
        if not AWS_IMAGE_URI:
            raise ValueError("AWS_IMAGE_URI must be set when using aws_image deployment mode")
        return modal.Image.from_registry(AWS_IMAGE_URI, add_python="3.12")

    elif DEPLOYMENT_MODE == "code":
        # Original code-based image definition
        return (
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
                # Install python-dotenv for secrets
                "cd /app && uv add python-dotenv",
            )
        )

    else:
        raise ValueError(f"Invalid deployment mode: {DEPLOYMENT_MODE}. Must be 'code', 'dockerfile', or 'aws_image'")


# Get the image based on deployment mode
image = get_image()


# Create a Modal app
app = modal.App("genai-framework")


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60,  # 1 hour timeout
    gpu=None,  # Optional: Use GPU if needed
    scaledown_window=300,  # Keep container alive for 5 minutes
)
@modal.web_server(8000, startup_timeout=600)  # 10 minutes startup timeout
def streamlit_server():
    """Serve the Streamlit app directly."""
    from pathlib import Path

    sys.path.append("/app")
    os.chdir("/app")

    # Set environment variables
    os.environ["BLUEPRINT_CONFIG"] = "container"
    os.environ["PYTHONPATH"] = "."

    # Create data directories if they don't exist
    os.makedirs(f"{VOLUME_PATH}/llm_cache", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/vector_store", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/hf_models", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/kv_store", exist_ok=True)

    # Ensure required files exist
    streamlit_file = Path("genai_blueprint/main/streamlit.py")
    if not streamlit_file.exists():
        print(f"ERROR: Streamlit file not found at {streamlit_file.absolute()}")
        print(f"Current directory: {Path.cwd()}")
        print(f"Directory contents: {list(Path.cwd().iterdir())}")
        raise FileNotFoundError(f"Streamlit app file not found: {streamlit_file}")

    print(f"Starting Streamlit server from {Path.cwd()}")
    print(f"Streamlit file exists: {streamlit_file.exists()}")

    # Start Streamlit server with proper settings for Modal
    cmd = [
        "uv",
        "run",
        "streamlit",
        "run",
        str(streamlit_file),
        "--server.port",
        "8000",
        "--server.address",
        "0.0.0.0",  # Bind to all interfaces for Modal
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.runOnSave",
        "false",
        "--server.fileWatcherType",
        "none",
    ]

    print(f"Executing command: {' '.join(cmd)}")

    # Use exec to replace the current process with Streamlit
    # This ensures Streamlit runs as the main process and binds to port 8000
    os.execvp("uv", cmd)


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
