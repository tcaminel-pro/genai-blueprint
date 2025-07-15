# Script to deploy to Modal
# Created by Claude Sonnet-4

# WIP - DOES NOT FULLY WORK (yet?) 


##############
##  MODAL  ###
##############

MODAL_ENTRY_POINT="src/main/modal_app.py"

.PHONY: modal_install modal_login modal_deploy modal_deploy_force modal_run modal_secrets modal_clear_cache

modal_install:  ## Install Modal CLI
	uv pip install modal

modal_login:  ## Login to Modal
	modal token new

modal_deploy:  ## Deploy to Modal (code mode)
	MODAL_DEPLOYMENT_MODE=code modal deploy $(MODAL_ENTRY_POINT)

modal_deploy_dockerfile:  ## Deploy to Modal using Dockerfile
	MODAL_DEPLOYMENT_MODE=dockerfile modal deploy $(MODAL_ENTRY_POINT)

modal_deploy_aws:  ## Deploy to Modal using AWS ECR image
	MODAL_DEPLOYMENT_MODE=aws_image MODAL_AWS_IMAGE_URI=$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION) modal deploy $(MODAL_ENTRY_POINT)

modal_deploy_github:  ## Deploy to Modal using GitHub registry image                                                       
	MODAL_DEPLOYMENT_MODE=aws_image MODAL_AWS_IMAGE_URI=ghcr.io/$(shell git config user.name | tr '[:upper:]' '[:lower:]')/$(basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]'):latest modal deploy $(MODAL_ENTRY_POINT)
modal_serve: 
	MODAL_DEPLOYMENT_MODE=code modal serve $(MODAL_ENTRY_POINT)

modal_serve_dockerfile:  ## Serve Modal using Dockerfile
	MODAL_DEPLOYMENT_MODE=dockerfile modal serve $(MODAL_ENTRY_POINT)

modal_serve_aws:  ## Serve Modal using AWS ECR image
	MODAL_DEPLOYMENT_MODE=aws_image MODAL_AWS_IMAGE_URI=$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION) modal serve $(MODAL_ENTRY_POINT)

modal_deploy_force:  ## Deploy to Modal with forced image rebuild (code mode)
	modal image clear
	MODAL_DEPLOYMENT_MODE=code modal deploy $(MODAL_ENTRY_POINT)

modal_deploy_force_dockerfile:  ## Deploy to Modal with forced image rebuild (dockerfile mode)
	modal image clear
	MODAL_DEPLOYMENT_MODE=dockerfile modal deploy $(MODAL_ENTRY_POINT)

modal_deploy_force_aws:  ## Deploy to Modal with forced image rebuild (aws mode)
	modal image clear
	MODAL_DEPLOYMENT_MODE=aws_image MODAL_AWS_IMAGE_URI=$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION) modal deploy $(MODAL_ENTRY_POINT)

modal_run:  ## Run locally with Modal (code mode)
	MODAL_DEPLOYMENT_MODE=code modal run $(MODAL_ENTRY_POINT)

modal_run_dockerfile:  ## Run locally with Modal using Dockerfile
	MODAL_DEPLOYMENT_MODE=dockerfile modal run $(MODAL_ENTRY_POINT)

modal_run_aws:  ## Run locally with Modal using AWS ECR image
	MODAL_DEPLOYMENT_MODE=aws_image MODAL_AWS_IMAGE_URI=$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION) modal run $(MODAL_ENTRY_POINT)

modal_clear_cache:  ## Clear Modal image cache
	modal image clear

modal_secrets:  ## Create Modal secrets from .env file
	modal secret create genai-secrets $$(cat .env | xargs)

modal_aws_secrets:  ## Create AWS credentials secret for ECR access
	@echo "Creating AWS credentials secret for Modal..."
	@if [ -z "$$AWS_ACCESS_KEY_ID" ] || [ -z "$$AWS_SECRET_ACCESS_KEY" ]; then \
		echo "Error: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set"; \
		echo "Run: export AWS_ACCESS_KEY_ID=your_key && export AWS_SECRET_ACCESS_KEY=your_secret"; \
		exit 1; \
	fi
	modal secret create aws-credentials \
		AWS_ACCESS_KEY_ID=$$AWS_ACCESS_KEY_ID \
		AWS_SECRET_ACCESS_KEY=$$AWS_SECRET_ACCESS_KEY \
		AWS_DEFAULT_REGION=$${AWS_DEFAULT_REGION:-eu-west-1}
