
APP=genai-blueprint
IMAGE_VERSION=0.2a

.PHONY: docker_build docker_run docker_shell

docker_build: ## Build the docker image
	docker build --pull --rm -f "deploy/Dockerfile" -t $(APP):$(IMAGE_VERSION) "."

docker_run: ## Run the container with environment variables
	@if [ ! -f .env ]; then \
		echo "Warning: .env file not found. Running without environment variables."; \
		docker run -it -p 8501:8501 $(APP):$(IMAGE_VERSION); \
	else \
		echo "Loading environment variables from .env file"; \
		docker run -it -p 8501:8501 \
			--env-file .env \
			$(APP):$(IMAGE_VERSION); \
	fi

docker_shell: ## Open a shell in the container
	@CONTAINER_ID=$$(docker ps --filter "ancestor=$(APP):$(IMAGE_VERSION)" --format "{{.ID}}" | head -1); \
	if [ -n "$$CONTAINER_ID" ]; then \
		docker exec -it $$CONTAINER_ID /bin/bash; \
	else \
		docker run -it --rm $(APP):$(IMAGE_VERSION) /bin/bash; \
	fi

