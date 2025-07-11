
#cSpell: disable
APP=genai-blueprint

MODAL_ENTRY_POINT="src/main/modal_app.py"
IMAGE_VERSION=0.2a

##############################
##  Build Docker, and run locally
##############################


test1:
	@echo "OpenAI Key:"  $(OPENAI_API_KEY)
	@echo ".env File:" $(ENV_FILE)

.PHONY: build run save sync_time check # Docker build and deployment 


docker_images: ## Check if the image is built
	docker images -a

sync_time:  # Needed because WSL loose time after hibernation, and that can cause issues when pushing 
	sudo hwclock -s 

docker_build: ## Build the docker image
	docker build --pull --rm -f "deploy/Dockerfile" -t $(APP):$(IMAGE_VERSION) "."

docker_run: ## Execute the image with environment variables
	docker run -it -p 8000:8000 -p 8501:8501 -p 443:443 \
		-e OPENROUTER_API_KEY=$(OPENROUTER_API_KEY) \
		-e DEEPSEEK_API_KEY=$(DEEPSEEK_API_KEY) \
		$(APP):$(IMAGE_VERSION)

docker_save:  # Create a zipped version of the image
	docker save $(APP):$(IMAGE_VERSION)| gzip > /tmp/$(APP)_$(IMAGE_VERSION).tar.gz

docker_shell: ## Open a shell in the running Docker container
	@echo "Opening shell in running container..."
	@CONTAINER_ID=$$(docker ps --filter "ancestor=$(APP):$(IMAGE_VERSION)" --format "{{.ID}}" | head -1); \
	if [ -n "$$CONTAINER_ID" ]; then \
		echo "Found running container: $$CONTAINER_ID"; \
		docker exec -it $$CONTAINER_ID /bin/bash; \
	else \
		echo "No running container found. Starting a new one with shell..."; \
		docker run -it --rm \
			-e OPENROUTER_API_KEY=$(OPENROUTER_API_KEY) \
			-e DEEPSEEK_API_KEY=$(DEEPSEEK_API_KEY) \
			$(APP):$(IMAGE_VERSION) /bin/bash; \
	fi

docker_shell_new: ## Start a new container with shell access
	@echo "Starting new container with shell..."
	docker run -it --rm \
		-e OPENROUTER_API_KEY=$(OPENROUTER_API_KEY) \
		-e DEEPSEEK_API_KEY=$(DEEPSEEK_API_KEY) \
		$(APP):$(IMAGE_VERSION) /bin/bash

docker_debug: ## Debug the Docker container environment
	@echo "=== Docker Container Debug ==="
	@CONTAINER_ID=$$(docker ps --filter "ancestor=$(APP):$(IMAGE_VERSION)" --format "{{.ID}}" | head -1); \
	if [ -n "$$CONTAINER_ID" ]; then \
		echo "Container ID: $$CONTAINER_ID"; \
		echo ""; \
		echo "=== Environment Variables ==="; \
		docker exec $$CONTAINER_ID env | grep -E "(API_KEY|TOKEN)" | sort; \
		echo ""; \
		echo "=== Running Processes ==="; \
		docker exec $$CONTAINER_ID ps aux; \
		echo ""; \
		echo "=== Network Ports ==="; \
		docker exec $$CONTAINER_ID netstat -tlnp 2>/dev/null || echo "netstat not available"; \
	else \
		echo "No running container found"; \
	fi

