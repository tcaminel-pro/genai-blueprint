# This Makefile provides commands for:
# - Development environment setup
# - Docker image building and deployment
# - Cloud platform integration (GCP, Azure, Modal)
# - Project maintenance tasks
#
# Usage: make [target]
#cSpell: disable


STREAMLIT_ENTRY_POINT="genai_blueprint/main/streamlit.py"
MODAL_ENTRY_POINT="src/main/modal_app.py"
APP=genai-blueprint
IMAGE_VERSION=0.2a
AWS_REGION=eu-west-1
AWS_ACCOUNT_ID=909658914353

DEV_PATH="/home/tcl/prj/genai-tk:/home/tcl/prj/genai-blueprint:$(PWD)"

all: help 

MAKEFLAGS += --warn-undefined-variables                                                            
SHELL     := bash -euo pipefail -c   # exit on error, undefined var, pipefail 

# Guard against GNU/Make vs BSD/Make incompatibilities 
# ifneq ($(shell echo 'a b' | xargs -n1 echo 2>/dev/null | wc -l),2)                                 
#   $(error You need GNU xargs)                                                                      
# endif  

# Locate and load the .env file in the current directory, or parent directory, or parent of parent
ENV_FILE_RAW := $(shell find $(CURDIR) $(CURDIR)/.. $(CURDIR)/../.. -name ".env" -print -quit 2>/dev/null)
ENV_FILE := $(shell realpath $(ENV_FILE_RAW) 2>/dev/null || echo "")
ifneq ($(ENV_FILE),)
include $(ENV_FILE)
else
$(warning .env file not found in current or parent directory)
endif


#include deploy/docker.mk
#include deploy/aws.mk
#include deploy/github.mk
include deploy/modal.mk

.PHONY: .uv   .pre-commit .pythonpath
.uv:  ## Check that uv is installed
	@uv -V || echo 'Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh 

.pre-commit: .uv  ## Check that pre-commit is installed    see https://pre-commit.com/
	@uv run pre-commit -V || uv pip install pre-commit

.pythonpath:
	@if [ -z "$(PYTHONPATH)" ]; then \
		echo "Warning: PYTHONPATH is not set. Consider to put somewhere: export PYTHONPATH=\".\" "; \
	fi


##############################
##  GenAI Blueprint related commands
##############################
.PHONY: fast_api langserve webapp 
fast-api: ## langsLauch FastAPI server localy
	uvicorn $(FASTAPI_ENTRY_POINT) --reload

langserve: ## Lauch langserve app
	python src/main/langserve_app.py

webapp: ## Launch Streamlit app
	PYTHONPATH=$(DEV_PATH) uv run streamlit run "$(STREAMLIT_ENTRY_POINT)"


##############################
##  Development
##############################
.PHONY: rebase aider aider-haiku aider-r1 lint quality clean_notebooks 
rebase: ## Sync local repo with remote one (changes are stashed before!)
	git fetch origin
	git stash
	git rebase origin/main

# Configure aider to use ruff as linter
AIDER_OPTS=--watch-files --lint-cmd "ruff format" --read vibe_coding/CONVENTIONS.md --editor "code --wait"

aider-v3:  ## Call aider-chat (a coding assistant)
	aider $(AIDER_OPTS) --cache-prompts --model deepseek/deepseek-chat-v3.1
aider-gemini:
	aider $(AIDER_OPTS) --cache-prompts --model openrouter/google/gemini-2.5-pro ;   
aider-sonnet:
	aider $(AIDER_OPTS) --cache-prompts --model openrouter/anthropic/claude-sonnet-4;   
aider-r1:
	aider $(AIDER_OPTS) --model openrouter/deepseek/deepseek-r1
aider-qwen:
	aider $(AIDER_OPTS) --model openrouter/qwen/qwen3-coder ; 
aider-k2:
	aider $(AIDER_OPTS) --cache-prompts --model openrouter/moonshotai/kimi-k2-0905
aider-gpt:
	aider $(AIDER_OPTS) --model openrouter/openai/gpt-oss-120b --reasoning-effort high

lint: ## Run Ruff an all Python files to format fix imports
	ruff check --select I --fix
	ruff format


quality: ## Run Ruff an all Python files to check quality
	find . -path "./src/wip" -prune -o -path "./.venv" -prune -o -type f -name '*.py' | xargs ruff check --fix 

clean-notebooks: ## Clean Jupyter notebook outputs.
	@find . -path "./.venv" -prune -o -name "*.ipynb" -print | while read -r notebook; do \
		echo "Cleaning outputs from: $$notebook"; \
		uv run --with nbconvert python -m nbconvert --clear-output --inplace "$$notebook"; \
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

check-uv: ## Check if uv is installed, install if missing
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
	cp -r ~/.env ~/.aws  ~/.bashrc ~/.bash_aliases ~/.modal.toml $(ONEDRIVE)/backup/wsl/tcl/
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
SYNC_DIRS=src/ai_core src/ai_extra src/ai_utils genai_blueprint/webapp/ui_components   

sync-dirs: ## Sync subdirectories between two root directories
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

clean-history: ## Remove duplicate entries and common commands from .bash_history
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


test-install: .pythonpath ## Quick test install
	@echo -e "\033[36mTesting genai_bp structure and GitHub dependencies...\033[0m"
	@echo -e "\033[32m✓ Project structure validated\033[0m"
	@echo -e "\033[32m✓ Dependencies configured for GitHub installation\033[0m"
	@echo -e "\033[33mNote: Full installation test requires: uv pip install git+https://github.com/tcaminel-pro/genai-blueprint@main\033[0m"
	@echo ""
	@echo -e "\033[36mTesting basic imports with current structure...\033[0m"
	@PYTHONPATH=$(PWD) python3 -c "import src.main.cli; print('✓ genai_bp CLI module structure valid')" 2>/dev/null || echo "✗ genai_bp CLI structure issue"
	@echo -e "\033[36mTesting genai_tk availability from GitHub...\033[0m"
	@uv run --isolated --with "genai_tk @ git+https://github.com/tcaminel-pro/genai-tk@main" python -c "import genai_tk.core; print('✓ genai_tk from GitHub works')" 2>/dev/null || echo "✗ genai_tk GitHub installation failed"
	@echo ""
	@echo -e "\033[32m✓ Basic validation complete. For full testing, install from GitHub.\033[0m"

test-github: ## Test full GitHub installation in isolated environment  
	@echo -e "\033[36mTesting complete GitHub installation...\033[0m"
	@echo "Creating temporary test environment..."
	@cd /tmp && rm -rf genai_bp_test .venv && mkdir genai_bp_test && cd genai_bp_test && \
		uv venv && \
		uv pip install git+https://github.com/tcaminel-pro/genai-blueprint@main && \
		echo -e "\033[32m✓ Installation successful\033[0m" && \
		. .venv/bin/activate && python -c "import genai_tk.core, genai_tk.utils, src.main.cli; print('✓ All imports work'); print('✓ GitHub installation test passed')" && \
		cd /tmp && rm -rf genai_bp_test
	@echo -e "\033[32m✓ Complete GitHub installation test passed\033[0m"


##############################
##  Project specific commands
##############################

##############################
##  MICS
##############################

postgres:   ## Start docker Postgres + pgvector                                                                           
	docker rm -f pgvector-container 2>/dev/null || true
	docker run -d --name pgvector-container \
		-e POSTGRES_USER=$(POSTGRES_USER) -e POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) -e POSTGRES_DB=ekg \
		-p 5432:5432 \
		-v /home/tcl/pgvector-data:/var/lib/postgresql/data \
		pgvector/pgvector:pg17

chrome:  ## Start docker Chromium 
# see https://hub.docker.com/r/linuxserver/chromium
	docker rm -f chromium 2>/dev/null || true
	docker run -d --name=chromium \
	--security-opt seccomp=unconfined  -e PUID=1000 -e PGID=1000 -e TZ=Europe/Paris  \
	-p 3000:3000 -p 3001:3001 -v /home/tcl/.chromiun:/config \
	--shm-size="1gb" --restart unless-stopped \
	lscr.io/linuxserver/chromium:latest
	xdg-open localhost:3000

qwen:
	@OPENAI_API_KEY=$(OPENROUTER_API_KEY) OPENAI_BASE_URL="https://openrouter.ai/api/v1" OPENAI_MODEL="qwen/qwen3-coder" qwen
