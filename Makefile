# This Makefile provides commands for:
# - Development environment setup
# - Docker image building and deployment
# - Cloud platform integration (GCP, Azure)
# - Project maintenance tasks
#
# Usage: make [target]


######################################
##  Development Environment ##
######################################

# General settings, notably for deployment

APP=genai-blueprint
STREAMLIT_ENTRY_POINT="python/GenAI_Lab.py"

FASTAPI_ENTRY_POINT="python.fastapi_app:app"

IMAGE_VERSION=0.2a
REGISTRY_AZ=XXXX.azurecr.io
REGISTRY_NAME=XXX
LOCATION=europe-west4
PROJECT_ID_GCP=XXX



.PHONY: telemetry 
.PHONY: clean lint clean_notebooks quality latest # Maintenance 
.PHONY: check_poetry install # Poetry and installation 
.PHONY: build run save sync_time check # Docker build and deployment 
.PHONY: login_gcp build_gcp push_gcp create_repo_gcp # GCP targets
.PHONY: push_az # Azure targets
.PHONY: backup  # Misc

##############################
##  GenAI Blueprint related commands
##############################
.PHONY: fast_api langserve webapp test rebase aider aider_haiku aider_r1  
fast_api:  # run Python code localy
	uvicorn $(FASTAPI_ENTRY_POINT) --reload

langserve:
	python python/langserve_app.py

webapp:
	streamlit run $(STREAMLIT_ENTRY_POINT)

##############################
##  Development
##############################
.PHONY: rebase aider aider_haiku aider_r1
.PHONY: lint quality clean_notebooks latest
rebase:
	git fetch origin
	git stash
	git rebase origin/main

AIDER_OPTS=--watch-files --no-auto-lint --read CONVENTIONS.md --editor nano
aider:
	aider $(AIDER_OPTS) --model deepseek/deepseek-chat; 

# not sure --cache-prompts works
aider_haiku: 
	aider $(AIDER_OPTS) / --cache-prompts --model openrouter/anthropic/claude-3-5-haiku;   
aider_r1:
	aider $(AIDER_OPTS) --model deepseek/deepseek-reasoner; 


lint:
	poetry run ruff check --select I --fix
	poetry run ruff format

quality:
	find . -path "./python/wip" -prune -o -type f -name '*.py' | xargs ruff check --fix

clean_notebooks:  ## Clean Jupyter notebook outputs. Require 'nbconvert' Python module
	@find . -name "*.ipynb" | while read -r notebook; do \
		echo "Cleaning outputs from: $$notebook"; \
		jupyter-nbconvert --clear-output --inplace "$$notebook"; \
	done

latest:  # Update selected fast changing dependencies 
	poetry add 	langchain@latest  langchain-core@latest langgraph@latest langserve@latest langchainhub@latest \
				 langchain-experimental@latest   langchain-community@latest  \
				 langchain-chroma@latest
	poetry add  gpt-researcher@latest browser-use@latest smolagents@latest mcpadapt@latest  --group ai_extra
#	poetry add crewai@latest[tools] --group demos



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
##  Poetry and project  intall
##############################

.PHONY:  check_poetry  install
check_poetry:  ## Check if poetry is installed, install if missing
	@command -v poetry >/dev/null 2>&1 || { \
		echo "Poetry is not installed. Installing now..."; \
		curl -sSL https://install.python-poetry.org | python3 -; \
		echo "Poetry installed successfully.\n Installing 'shell'  plugin..."; \
		poetry self add poetry-plugin-shell; \
		echo "Plugin installed ""; \
	}

install: check_poetry  ## Install project core dependencies
	poetry lock
	poetry install --without ai_extra, demos, transformers, instrumentation

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

build_gcp: ## build the image
	docker build -t gcr.io/$(PROJECT_ID_GCP)/$(APP):$(IMAGE_VERSION) . --build-arg OPENAI_API=$(OPENAI_API_KEY) 

push_gcp:
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
	
push_az:  # Push to a registry
	docker tag $(APP):$(IMAGE_VERSION) $(REGISTRY_AZ)/$(APP):$(IMAGE_VERSION)
	docker push $(REGISTRY_AZ)/$(APP):$(IMAGE_VERSION)

# To be completed...


##############
##  MISC  ###
##############

.PHONY: backup  clean dedupe_history

clean:  ## Clean Python bytecode and cache files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ruff_cache" -delete
	find . -type d -name ".mypy_cache" -delete


backup:
# copy to ln_to_onedrive, a symbolic link from WSL to OneDrive 
# (created as: ln -s '/mnt/c/Users/a184094/OneDrive - Eviden'  ~/ln_to_onedrive )
	cp ~/.env ~/.bashrc ~/.dev.bash-profile ~/ln_to_onedrive/backup/wsl/tcl/
	cp ~/install.sh  ~/ln_to_onedrive/backup/wsl/tcl/

	rsync -av \
	--exclude='.git/' --exclude='.ruf_cache/' --exclude='__pycache__/'  \
	--include='*/' \
	--include='*.py' --include='*.ipynb' --include='*.toml' --include='*.yaml' --include='*.json' \
	--include='Makefile' --include='Dockerfile' \
	--exclude='*' \
	~/prj ~/ln_to_onedrive/backup/wsl/tcl


dedupe_history:  ## Remove duplicate entries and common commands from .bash_history while preserving order
	@if [ -f ~/.bash_history ]; then \
		awk '!/^(ls|cat|hgrep)( |$$)/ && !seen[$$0]++' ~/.bash_history > ~/.bash_history_unique && \
		mv ~/.bash_history_unique ~/.bash_history; \
		echo "Done : duplicates and common commands removed. \nRun 'history -c; history -r' in your shell to reload the cleaned history"; \
	else \
		echo "No .bash_history file found"; \
	fi



##############################
##  Project specific commands
##############################


