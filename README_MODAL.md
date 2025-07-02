# Modal Deployment for GenAI Framework

This document explains how to deploy the GenAI Framework Streamlit application to Modal.

## Prerequisites

1. Install Modal CLI:
```bash
make modal_install
```
or
```bash
uv pip install modal
```

2. Set up Modal account:
```bash
make modal_login
```
or
```bash
modal token new
```

## Deployment Steps

### 1. Configure your secrets in Modal

You can create secrets directly from your .env file:

```bash
make modal_secrets
```
or
```bash
modal secret create genai-secrets $(cat .env | xargs)
```

### 2. Deploy the application

```bash
make modal_deploy
```
or
```bash
modal deploy src/main/modal_app.py
```

### 3. Run the app locally with Modal

```bash
make modal_run
```
or
```bash
modal run src/main/modal_app.py
```

## Configuration

The deployment uses a Modal-specific configuration that:

1. Sets data paths to use the Modal volume at `/data`
2. Configures OpenAI models as the default
3. Disables authentication for easier access

You can modify these settings in:
- `config/modal.yaml` - Modal-specific configuration file
- `config/overrides.yaml` - Contains the `modal` configuration section

## Environment Variables

The following environment variables should be configured in your Modal secrets:

- `OPENAI_API_KEY`: Your OpenAI API key
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
- `GROQ_API_KEY`: Your Groq API key
- `LANGCHAIN_API_KEY`: Your LangChain API key
- `OPENROUTER_API_KEY`: Your OpenRouter API key

Add any other API keys or configuration variables from your .env file.

## Volumes

The application uses a Modal volume named `genai-data` to persist data between runs. This volume is mounted at `/data` and contains:

- `/data/llm_cache` - LLM cache files
- `/data/vector_store` - Vector store data
- `/data/hf_models` - Hugging Face models
- `/data/kv_store` - Key-value store data

## Advanced Configuration

### GPU Support

The deployment is configured to use a GPU if available. To specify a particular GPU type:

```python
@stub.function(
    # ...
    gpu="T4",  # or "A100", etc.
)
```

### Custom Dependencies

If you need additional dependencies, modify the image definition in `modal_app.py`.

### Scheduled Runs

To schedule periodic runs of your application:

```python
@stub.function(
    # ...
    schedule=modal.Period(days=1),  # Run daily
)
```

## Updating the Image

When you make changes to your source code, you don't need to redeploy the entire application. Modal provides several ways to update just the image:

### Force Image Rebuild

To rebuild the image with your latest source changes:

```bash
make modal_deploy_force
```
or
```bash
modal deploy src/main/modal_app.py --force-build
```

### Clear Image Cache

If you want to clear the cached image and rebuild from scratch:

```bash
modal image clear
```

Then redeploy:

```bash
modal deploy src/main/modal_app.py
```

### Development Workflow

For faster iteration during development, you can use:

```bash
modal run src/main/modal_app.py --detach
```

This runs the app in detached mode and automatically picks up source changes.

## Troubleshooting

- If you encounter dependency issues, check the Modal logs for details.
- For file access issues, ensure paths are correctly configured to use the Modal volume.
- If authentication is needed, set `auth.enabled: true` in the Modal configuration.
- If your changes aren't reflected, try using `--force-build` to rebuild the image.

## Modal 1.0 Migration Notes

This deployment has been updated to be compatible with Modal 1.0:

1. Replaced `modal.Mount` with `image.add_local_dir` for better performance
2. Updated volume mounting to use the `volumes` parameter
3. Changed `modal.App` to `modal.Stub` for consistency

For more information on Modal 1.0 changes, see the [Modal 1.0 migration guide](https://modal.com/docs/guide/modal-1-0-migration).
