
##############
##  MODAL  ###
##############

.PHONY: modal_install modal_login modal_deploy modal_deploy_force modal_run modal_secrets modal_clear_cache

modal_install:  ## Install Modal CLI
	uv pip install modal

modal_login:  ## Login to Modal
	modal token new

modal_deploy:  ## Deploy to Modal
	modal deploy $(MODAL_ENTRY_POINT)

modal_serve: 
	modal serve $(MODAL_ENTRY_POINT)

modal_deploy_force:  ## Deploy to Modal with forced image rebuild
	modal image clear
	modal deploy $(MODAL_ENTRY_POINT)

modal_run:  ## Run locally with Modal
	modal run $(MODAL_ENTRY_POINT)

modal_clear_cache:  ## Clear Modal image cache
	modal image clear

modal_secrets:  ## Create Modal secrets from .env file
	modal secret create genai-secrets $$(cat .env | xargs)
