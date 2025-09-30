
# APP and IMAGE_VERSION are defined in main Makefile

.PHONY: docker_build docker_run docker_shell

docker-build: ## Build the docker image
	docker build --pull --rm -f "deploy/Dockerfile" -t $(APP):$(IMAGE_VERSION) "."

docker-run: ## Run the container with environment variables and mounted training data
	@echo "Loading environment variables from .env file"; \
	echo "Mounting external data from $(ONEDRIVE)/_ongoing/training_GenAI/"; \
	docker run -it -p 8501:8501 \
		--env-file $(ENV_FILE) \
		-e REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
		-e SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
		-e CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
		-v "$(ONEDRIVE)/_ongoing/training_GenAI/:/data/external" \
		$(APP):$(IMAGE_VERSION); 



