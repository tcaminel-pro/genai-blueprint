# This Makefile provides commands for:
# - Development environment setup
# - Docker image building and deployment
# - Cloud platform integration (GCP, Azure)
# - Project maintenance tasks
#
# Usage: make [target]


##############################
##  Development Environment ##
##############################

# General settings, notably for deployment

APP=genai-blueprint
STREAMLIT_ENTRY_POINT="python/GenAI_Lab.py"

FASTAPI_ENTRY_POINT="python.fastapi_app:app"

IMAGE_VERSION=0.2a
REGISTRY_AZ=XXXX.azurecr.io
REGISTRY_NAME=XXX
LOCATION=europe-west4
PROJECT_ID_GCP=XXX


.PHONY: check fast_api langserve webapp test rebase aider \
        import_files sync_time build run save \
        login_gcp build_gcp push_gcp create_repo_gcp \
        push_az latest clean lint backup


######################
##  GenAI Blueprint related commands
#####################

check: ## Check if the image is built
	docker images -a

fast_api:  # run Python code localy
	uvicorn $(FASTAPI_ENTRY_POINT) --reload

langserve:
	python python/langserve_app.py

webapp:
	streamlit run $(STREAMLIT_ENTRY_POINT)

test:
	pytest -s

rebase:
	git fetch origin
	git stash
	git rebase origin/main


AIDER_OPTS=--watch-files --no-auto-lint --read CONVENTIONS.md --editor nano
aider:  ## launch aider-chat (a coding assistant) with our configuration. 
	if [ "$(filter haiku,$(MAKECMDGOALS))" ]; then \
		aider $(AIDER_OPTS) --cache-prompts --model openrouter/anthropic/claude-3-5-haiku; \
	else \
		aider $(AIDER_OPTS) --model deepseek/deepseek-chat; \
	fi


######################
##  Project build commands
#####################



######################
##  Build Docker, and run locally
#####################

#WARNING : Put the API key into the docker image. NOT RECOMMANDED IN PRODUCTION

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

# To be commeted...

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

	
push_az:  # Push to a registry
	docker tag $(APP):$(IMAGE_VERSION) $(REGISTRY_AZ)/$(APP):$(IMAGE_VERSION)
	docker push $(REGISTRY_AZ)/$(APP):$(IMAGE_VERSION)

# To be completed...


##############
##  MISC  ###
##############

clean_notebooks:  ## Clean Jupyter notebook outputs
	@if ! command -v jupyter-nbconvert &> /dev/null; then \
		echo "jupyter-nbconvert could not be found. Please install it first:"; \
		echo "pip install nbconvert"; \
		exit 1; \
	fi
	@find . -name "*.ipynb" | while read -r notebook; do \
		echo "Cleaning outputs from: $$notebook"; \
		jupyter-nbconvert --clear-output --inplace "$$notebook"; \
	done
	@echo "Notebook outputs cleaned successfully!"

latest:  # Update selected fast changing dependencies 
	poetry add 	langchain@latest  langchain-core@latest langgraph@latest langserve@latest langchainhub@latest \
				 langchain-experimental@latest   langchain-community@latest  \
				 langchain-chroma@latest
	poetry add  gpt-researcher@latest browser-use@latest smolagents@latest   langchain-mcp@latest   --G ai_extra
	poetry add crewai@latest[tools] -G demos

# aider-chat@latest
# litellm@latest lunary@
#langchain-openai@latest
# langchain-groq@latest    \


clean:  ## Clean Python bytecode and cache files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ruff_cache" -delete
	find . -type d -name ".mypy_cache" -delete

lint:
	poetry run ruff check --select I --fix
	poetry run ruff format

backup:
# copy to ln_to_onedrive, a symbolic link from WSL to OneDrive 
# (created as: ln -s '/mnt/c/Users/a184094/OneDrive - Eviden'  ~/ln_to_onedrive )
	cp ~/.{env,bashrc,dev.bash-profile,bash_history_unique} ~/ln_to_onedrive/backup/wsl/tcl/
	cp ~/install.sh  ~/ln_to_onedrive/backup/wsl/tcl/

	rsync -av \
	--exclude='.git/' --exclude='.ruf_cache/' --exclude='__pycache__/'  \
	--include='*/' \
	--include='*.py' --include='*.ipynb' --include='*.toml' --include='*.yaml' --include='*.json' \
	--include='Makefile' --include='Dockerfile' \
	--exclude='*' \
	~/prj ~/ln_to_onedrive/backup/wsl/tcl




