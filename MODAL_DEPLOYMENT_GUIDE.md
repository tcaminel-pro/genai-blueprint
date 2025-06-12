# Deploying Streamlit + LLM App to Modal from GitHub

## Option 1: Basic CLI Deployment
```python
import modal

stub = modal.Stub("genai-blueprint")

image = (
    modal.Image.debian_slim()
    .pip_install("streamlit")
    .run_commands(
        "git clone https://github.com/yourusername/genai-blueprint /app",
        "cd /app && pip install -e .",
    )
)

@stub.function(image=image, gpu="any")
def run_app():
    import subprocess
    subprocess.run(["streamlit", "run", "/app/src/main/streamlit.py"])
```

## Option 2: GitHub Integration (Recommended)
1. Connect GitHub account in Modal dashboard
2. Add `modal.yaml` to repo:
```yaml
app:
  name: genai-blueprint
  gpu: any
  timeout: 3600
```

## Option 3: Private Repo Setup
```python
image = (
    modal.Image.debian_slim()
    .pip_install("streamlit")
    .run_commands(
        "git clone git@github.com:yourusername/private-repo.git /app",
        "cd /app && pip install -e .[ai,ui]",
        secrets=[modal.Secret.from_name("github-ssh-key")],
    )
)
```

## Key Configuration Options
| Setting          | Recommendation           |
|------------------|--------------------------|
| GPU              | `gpu="a10g"` for LLMs    |
| Timeout          | `timeout=3600` (1 hour)  |
| Secrets          | Store API keys in Modal  |
| Dependencies     | Install from pyproject.toml |

## Deployment Commands
```bash
# First time setup
modal setup

# Deploy
modal deploy main.py

# Run temporarily
modal run main.py::stub.run_app
```

## Post-Deployment
1. Access via Modal-provided URL
2. Monitor logs in Modal dashboard
3. Configure auto-scaling if needed
