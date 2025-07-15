
APP=genai-blueprint
IMAGE_VERSION=0.2a

.PHONY: docker_build docker_run docker_shell

docker_build: ## Build the docker image
	docker build --pull --rm -f "deploy/Dockerfile" -t $(APP):$(IMAGE_VERSION) "."

docker_run: ## Run the container with environment variables
	@echo "Loading environment variables from .env file"; \
	docker run -it -p 8501:8501 \
		--env-file $(ENV_FILE) \
		-e REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
		-e SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
		-e CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
		$(APP):$(IMAGE_VERSION)

docker_shell: ## Open a shell in the container
	@CONTAINER_ID=$$(docker ps --filter "ancestor=$(APP):$(IMAGE_VERSION)" --format "{{.ID}}" | head -1); \
	if [ -n "$$CONTAINER_ID" ]; then \
		docker exec -it $$CONTAINER_ID /bin/bash; \
	else \
		if [ -z "$(ONEDRIVE)" ]; then \
			echo "Warning: ONEDRIVE environment variable not set. Running without training data mount."; \
			docker run -it --rm $(APP):$(IMAGE_VERSION) /bin/bash; \
		else \
			echo "Mounting training data from $(ONEDRIVE)/_ongoing/training_GenAI/"; \
			docker run -it --rm \
				-v "$(ONEDRIVE)/_ongoing/training_GenAI/:/data/external" \
				$(APP):$(IMAGE_VERSION) /bin/bash; \
		fi; \
	fi

