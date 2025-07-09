
##############
##  MODAL  ###
##############

.PHONY: modal_install modal_login modal_deploy modal_deploy_force modal_run modal_secrets modal_clear_cache

modal_install:  ## Install Modal CLI
	uv pip install modal

modal_login:  ## Login to Modal
	modal token new

modal_deploy:  ## Deploy to Modal (code mode)
	MODAL_DEPLOYMENT_MODE=code modal deploy $(MODAL_ENTRY_POINT)

modal_deploy_dockerfile:  ## Deploy to Modal using Dockerfile
	MODAL_DEPLOYMENT_MODE=dockerfile modal deploy $(MODAL_ENTRY_POINT)

modal_deploy_aws:  ## Deploy to Modal using AWS image (set MODAL_AWS_IMAGE_URI)
	MODAL_DEPLOYMENT_MODE=aws_image modal deploy $(MODAL_ENTRY_POINT)

modal_deploy_github:  ## Deploy to Modal using GitHub registry image                                                       
    MODAL_DEPLOYMENT_MODE=aws_image MODAL_AWS_IMAGE_URI=ghcr.io/$(shell git config user.name | tr '[:upper:]' '[:lower:]')/$(basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]'):latest modal deploy $(MODAL_ENTRY_POINT)
modal_serve: 
	MODAL_DEPLOYMENT_MODE=code modal serve $(MODAL_ENTRY_POINT)

modal_serve_dockerfile:  ## Serve Modal using Dockerfile
	MODAL_DEPLOYMENT_MODE=dockerfile modal serve $(MODAL_ENTRY_POINT)

modal_serve_aws:  ## Serve Modal using AWS image (set MODAL_AWS_IMAGE_URI)
	MODAL_DEPLOYMENT_MODE=aws_image modal serve $(MODAL_ENTRY_POINT)

modal_deploy_force:  ## Deploy to Modal with forced image rebuild (code mode)
	modal image clear
	MODAL_DEPLOYMENT_MODE=code modal deploy $(MODAL_ENTRY_POINT)

modal_deploy_force_dockerfile:  ## Deploy to Modal with forced image rebuild (dockerfile mode)
	modal image clear
	MODAL_DEPLOYMENT_MODE=dockerfile modal deploy $(MODAL_ENTRY_POINT)

modal_deploy_force_aws:  ## Deploy to Modal with forced image rebuild (aws mode)
	modal image clear
	MODAL_DEPLOYMENT_MODE=aws_image modal deploy $(MODAL_ENTRY_POINT)

modal_run:  ## Run locally with Modal (code mode)
	MODAL_DEPLOYMENT_MODE=code modal run $(MODAL_ENTRY_POINT)

modal_run_dockerfile:  ## Run locally with Modal using Dockerfile
	MODAL_DEPLOYMENT_MODE=dockerfile modal run $(MODAL_ENTRY_POINT)

modal_run_aws:  ## Run locally with Modal using AWS image (set MODAL_AWS_IMAGE_URI)
	MODAL_DEPLOYMENT_MODE=aws_image modal run $(MODAL_ENTRY_POINT)

modal_clear_cache:  ## Clear Modal image cache
	modal image clear

modal_secrets:  ## Create Modal secrets from .env file
	modal secret create genai-secrets $$(cat .env | xargs)
