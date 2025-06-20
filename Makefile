# This Makefile provides commands for:
# - Development environment setup
# - Docker image building and deployment
# - Cloud platform integration (GCP, Azure)
# - Project maintenance tasks
#
# Usage: make [target]
#cSpell: disable
APP=genai-blueprint

STREAMLIT_ENTRY_POINT="src/main/streamlit.py"
FASTAPI_ENTRY_POINT="src.main.fastapi_app:app"

IMAGE_VERSION=0.2a
REGISTRY_AZ=XXXX.azurecr.io
REGISTRY_NAME=XXX
LOCATION=europe-west4
PROJECT_ID_GCP=XXX

all: help 


SHELL := /bin/bash

# Locate and load the .env file in the current directory, or parent directory, or parent of parent
ENV_FILE_RAW := $(shell find $(CURDIR) $(CURDIR)/.. $(CURDIR)/../.. -name ".env" -print -quit)
ENV_FILE := $(shell realpath $(ENV_FILE_RAW))
ifneq ($(ENV_FILE),)
include $(ENV_FILE)
else
$(warning .env file not found in current or parent directory)
endif


.PHONY: .uv   .pre-commit .pythonpath
.uv:  ## Check that uv is installed
	@uv -V || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.pre-commit: .uv  ## Check that pre-commit is installed    see https://pre-commit.com/
	@uv run pre-commit -V || uv pip install pre-commit

.pythonpath:
	@if [ -z "$(PYTHONPATH)" ]; then \
		@echo "Warning: PYTHONPATH is not set. Consider to put somewhere: export PYTHONPATH=\".\" "; \
	fi


##############################
##  GenAI Blueprint related commands
##############################
.PHONY: fast_api langserve webapp 
fast_api:  ## langsLauch FastAPI server localy
	uvicorn $(FASTAPI_ENTRY_POINT) --reload

langserve: ## Lauch langserve app
	python src/main/langserve_app.py

webapp: ## Lauch Streamlit app
	uv run streamlit run $(STREAMLIT_ENTRY_POINT)


##############################
##  Development
##############################
.PHONY: rebase aider aider-haiku aider-r1 lint quality clean_notebooks 
rebase: ## Sync local repo with remote one (changes are stashed before!)
	git fetch origin
	git stash
	git rebase origin/main

# Configure aider to use ruff as linter
AIDER_OPTS=--watch-files --lint-cmd "ruff format" --read CONVENTIONS.md --editor "code --wait"

aider:  ## Call aider-chat (a coding assistant)
	aider $(AIDER_OPTS) --model openrouter/deepseek/deepseek-chat
aider-haiku: 
	aider $(AIDER_OPTS) --cache-prompts --model openrouter/anthropic/claude-3-5-haiku;   
aider-sonnet: 
	aider $(AIDER_OPTS) --cache-prompts --model openrouter/anthropic/claude-3.7-sonnet;   
aider-r1:
	aider $(AIDER_OPTS) --model openrouter/deepseek/deepseek-r1
aider-o3:
	aider $(AIDER_OPTS) --model o3-mini; 


lint: ## Run Ruff an all Python files to format fix imports
	ruff check --select I --fix
	ruff format


quality: ## Run Ruff an all Python files to check quality
	find . -path "./src/wip" -prune -o -path "./.venv" -prune -o -type f -name '*.py' | xargs ruff check --fix 

clean-notebooks:  ## Clean Jupyter notebook outputs. 
	@find . -name "*.ipynb" | while read -r notebook; do \
		echo "Cleaning outputs from: $$notebook"; \
		uvx jupyter nbconvert --clear-output --inplace "$$notebook"; \
	done

##############################
##  Telemetry  Tasks
##############################
.PHONY: telemetry

telemetry:  ## Run Phoenix telemetry server in background
	@echo "Starting Phoenix telemetry server..."
	@if ! pgrep -f "phoenix.server.main" > /dev/null; then \
		python -m phoenix.server.main serve > /tmp/phoenix.log 2>&1 & \
		echo "Phoenix server started in background (PID: $$!)"; \
		echo "Logs are being written to /tmp/phoenix.log"; \
		echo "look at: http://localhost:6006/projects" \
	else \
		echo "Phoenix server is already running"; \
	fi

##############################
##  uv and project  install
##############################

.PHONY: check_uv install

check_uv:  ## Check if uv is installed, install if missing
	@if command -v uv >/dev/null 2>&1; then \
		echo "uv is already installed"; \
	else \
		echo "uv is not installed. Installing now..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "uv installed successfully"; \
		. $HOME/.local/bin/env; \
	fi

install: check_uv   ## Install SW
	uv sync

install_spacy_models:  ## Install Spacy Models for NLP
	uv run --with spacy spacy download fr_core_news_sm
	uv run --with spacy spacy download en_core_web_lg


##############################
##  Build Docker, and run locally
##############################
#WARNING : Put the API key into the docker image. NOT RECOMMANDED IN PRODUCTION

.PHONY: build run save sync_time check # Docker build and deployment 


check: ## Check if the image is built
	docker images -a

sync_time:  # Needed because WSL loose time after hibernation, and that can cause issues when pushing 
	sudo hwclock -s 

build: ## build the docker image
	docker build --pull --rm -f "Dockerfile" -t $(APP):$(IMAGE_VERSION) "." \
      --build-arg OPENAI_API_KEY=$(OPENAI_API_KEY) \
	  --build-arg GROQ_API_KEY=$(GROQ_API_KEY) \
	  --build-arg LANGCHAIN_API_KEY=$(LANGCHAIN_API_KEY) 

run: ## execute the image locally
	docker run -it  -p 8000:8000 -p 8501:8501 $(APP):$(IMAGE_VERSION)

save:  # Create a zipped version of the image
	docker save $(APP):$(IMAGE_VERSION)| gzip > /tmp/$(APP)_$(IMAGE_VERSION).tar.gz


##############
##  GCP  ###
##############

.PHONY: login_gcp build_gcp push_gcp create_repo_gcp # GCP targets

# To be completed...

login_gcp:
	gcloud auth login
	gcloud config set project  $(PROJECT_ID_GCP)

build_gcp: ## build the image gor GCP
	docker build -t gcr.io/$(PROJECT_ID_GCP)/$(APP):$(IMAGE_VERSION) . --build-arg OPENAI_API=$(OPENAI_API_KEY) 

push_gcp: ## Push to a GCP registry
# gcloud auth configure-docker
	docker tag $(APP):$(IMAGE_VERSION) $(LOCATION)-docker.pkg.dev/$(PROJECT_ID_GCP)/$(REGISTRY_NAME)/$(APP):$(IMAGE_VERSION)
	docker push $(LOCATION)-docker.pkg.dev/$(PROJECT_ID_GCP)/$(REGISTRY_NAME)/$(APP):$(IMAGE_VERSION)
# gcloud run deploy --image gcr.io/$(PROJECT_ID_GCP)/$(APP):$(IMAGE_VERSION) --platform managed

create_repo_gcp:
	gcloud auth configure-docker $(LOCATION)-docker.pkg.dev
	gcloud artifacts repositories create $(REGISTRY_NAME) --repository-format=docker \
		--location=$(LOCATION) --description="Docker repository" \
		--project=$(PROJECT_ID_GCP)
		
##############
##  AZURE  ###
##############
.PHONY: push_az # Azure targets
	
push_az:  ## Push to a Azure registry
	docker tag $(APP):$(IMAGE_VERSION) $(REGISTRY_AZ)/$(APP):$(IMAGE_VERSION)
	docker push $(REGISTRY_AZ)/$(APP):$(IMAGE_VERSION)

# To be completed...


##############
##  MISC  ###
##############

.PHONY: backup  clean clean_history help

clean:  ## Clean Python bytecode and cache files
	uv cache prune
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ruff_cache" -delete
	find . -type d -name ".mypy_cache" -delete


backup: ## rsync project and shared files to $(ONEDRIVE)
# (created as: ln -s '/mnt/c/Users/a184094/OneDrive - Eviden'  ~/to_onedrive )
	cp ~/.env ~/.bashrc ~/.bash_aliases $(ONEDRIVE)/backup/wsl/tcl/
	cp ~/install.sh  $(ONEDRIVE)/backup/wsl/tcl/

backup-sync: 
	rsync -av \
	--exclude='.git/' --exclude='.ruf_cache/' --exclude='__pycache__/'  \
	--include='*/' \
	--include='*.py' --include='*.ipynb' --include='*.toml' --include='*.yaml' --include='*.json' \
	--include='Makefile' --include='Dockerfile' \
	--exclude='*' \
	~/prj $(ONEDRIVE)/backup/wsl/tcl


ROOT1=/home/tcl/prj/genai-blueprint/
ROOT2=/home/tcl/prj/ecod-engine-v3
SYNC_DIRS=src/ai_core src/ai_extra src/ai_utils src/webapp/ui_components   

sync_dirs: ## Sync subdirectories between two root directories 
	@if [ -z "$(ROOT1)" ] || [ -z "$(ROOT2)" ] || [ -z "$(SYNC_DIRS)" ]; then \
		echo "Error: Missing required variables. Usage: make sync_dirs ROOT1=path1 ROOT2=path2 SYNC_DIRS='dir1 dir2 dir3'"; \
		exit 1; \
	fi; \
	for dir in $(SYNC_DIRS); do \
		if [ -d "$(ROOT1)/$$dir" ] && [ -d "$(ROOT2)/$$dir" ]; then \
			echo "Synchronizing '$$dir' between '$(ROOT1)' and '$(ROOT2)''..."; \
			rsync -av --include='*/' --include='*.py' --update "$(ROOT1)/$$dir/" "$(ROOT2)/$$dir/"; \
			rsync -av --include='*/' --include='*.py' --update "$(ROOT2)/$$dir/" "$(ROOT1)/$$dir/"; \
		else \
			echo "Directory '$$dir' not found in both roots, skipping..."; \
		fi; \
	done

clean_history:  ## Remove duplicate entries and common commands from .bash_history
	@if [ -f ~/.bash_history ]; then \
		awk '!/^(ls|cat|hgrep|h|cd|p|m|ll|pwd|code|mkdir|export|rmdir|uv tree|make)( |$$)/ && !seen[$$0]++' ~/.bash_history > ~/.bash_history_unique && \
		mv ~/.bash_history_unique ~/.bash_history; \
		echo "Done : duplicates and common commands removed. \nRun 'history -c; history -r' in your shell to reload the cleaned history"; \
	else \
		echo "No .bash_history file found"; \
	fi                                                                                                  

help:                                                                                                                                          
	@echo "Available targets:"                                                                                                                  
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | sort 


test_install: .pythonpath ## Quick test install
	@echo $(PYTHONPATH)
	@echo "Call a fake LLM that returns the prompt. Here it should  display 'tell me a joke on ...'"
	echo bears | PYTHONPATH="." uv run cli run joke -m parrot_local_fake


# Load .env file environ variable in shell
# export $(grep -v '^#' ~/.env | xargs)

load_env:
	@echo "Loading environment variables..."
	@if [ -f ~/.env ]; then \
		export $$(grep -v '^#' ~/.env | xargs); \
	fi


env_path:
	echo $(ENV_FILE)

call-azure-llm: load_env
	curl -X POST "$(AZURE_OPENAI_ENDPOINT)/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-15-preview" \
	-H "Content-Type: application/json" \
	-H "Authorization: Bearer $(AZURE_OPENAI_API_KEY)" \
	-d "{\"messages\":[{\"role\":\"user\",\"content\":\"Tell me a joke\"}]}"

call-openrouter-llm: 
	@echo "Calling OpenRouter LLM..."
	curl https://openrouter.ai/api/v1/chat/completions \
	-H "Content-Type: application/json" \
	-H "Authorization: Bearer $(OPENROUTER_API_KEY)" \
	-d "{ \"model\": \"openai/gpt-4.1-mini\", \"messages\":[{\"role\":\"user\",\"content\":\"Tell me a joke\"}]}"

##############################
##  Project specific commands
##############################

##############################
##  MICS
##############################


