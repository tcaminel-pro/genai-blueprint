
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
	docker run -it -p 8000:8000 -p 8501:8501 \
		-e OPENROUTER_API_KEY=$(OPENROUTER_API_KEY) \
		-e DEEPSEEK_API_KEY=$(DEEPSEEK_API_KEY) \
		$(APP):$(IMAGE_VERSION)

docker_save:  # Create a zipped version of the image
	docker save $(APP):$(IMAGE_VERSION)| gzip > /tmp/$(APP)_$(IMAGE_VERSION).tar.gz

