
APP=genai-blueprint
IMAGE_VERSION=0.2a

.PHONY: docker_build docker_run docker_shell

docker_build: ## Build the docker image
	docker build --pull --rm -f "deploy/Dockerfile" -t $(APP):$(IMAGE_VERSION) "."

docker_run: ## Run the container
	docker run -it -p 8501:8501 \
		$(APP):$(IMAGE_VERSION)

docker_shell: ## Open a shell in the container
	@CONTAINER_ID=$$(docker ps --filter "ancestor=$(APP):$(IMAGE_VERSION)" --format "{{.ID}}" | head -1); \
	if [ -n "$$CONTAINER_ID" ]; then \
		docker exec -it $$CONTAINER_ID /bin/bash; \
	else \
		docker run -it --rm $(APP):$(IMAGE_VERSION) /bin/bash; \
	fi

