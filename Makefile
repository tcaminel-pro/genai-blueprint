# Makefile

export APP=genai-tcl
export IMAGE_VERSION=0.1a
export REGISTRY_AZ=XXXX.azurecr.io

export PROJECT_ID_GCP=nicolas-mathieu-0307-bine

export STREAMLIT_ENTRY_POINT="python/GenAI_Training.py"

#WARNING : Put the API key into the docker image. NOT RECOMMANDED IN PRODUCTION
build: ## build the image
	docker build --pull --rm -f "Dockerfile" -t $(APP):$(IMAGE_VERSION) "." \
      --build-arg OPENAI_API_KEY=$(OPENAI_API_KEY) \
	  --build-arg GROQ_API_KEY=$(GROQ_API_KEY) 

login_gcp:
	gcloud auth login

build_gcp: ## build the image
	docker build -t gcr.io/$(PROJECT_ID_GCP)/$(APP):$(IMAGE_VERSION) . --build-arg OPENAI_API=$(OPENAI_API_KEY) 

push_gcp:
	gcloud auth configure-docker
	docker push gcr.io/$(PROJECT_ID_GCP)/$(APP):$(IMAGE_VERSION)
	gcloud run deploy --image gcr.io/$(PROJECT_ID_GCP)/$(APP):$(IMAGE_VERSION) --platform managed

run: ## execute the image locally
	docker run -it  -p 8000:8000 -p 8501:8501 $(APP):$(IMAGE_VERSION)

check: ## Check if the image is built
	docker images -a

fast_api:  # run Python code localy
	uvicorn python.fastapi_app:app --reload

webapp:
	streamlit run $(STREAMLIT_ENTRY_POINT)


sync_time:  # Needed because WSL loose time after hibernation, and that can cause issues when pushing 
	sudo hwclock -s 

test:
	pytest -s
	
push_az:  # Push to a registry
	docker tag $(APP):$(IMAGE_VERSION) $(REGISTRY_AZ)/$(APP):$(IMAGE_VERSION)
	docker push $(REGISTRY_AZ)/$(APP):$(IMAGE_VERSION)

save:  # Create a zipped version of the image
	docker save $(APP):$(IMAGE_VERSION)| gzip > /tmp/$(APP)_$(IMAGE_VERSION).tar.gz

save_gcp:  # Create a zipped version of the image
	docker save $(APP):$(IMAGE_VERSION)| gzip > /tmp/$(APP)_$(IMAGE_VERSION).tar.gz

update:  # Update selected fast changing dependencies
	poetry add 	langchain@latest langchain-experimental@latest  langchain-core@latest  langchain-community@latest langgraph@latest langserve@latest langchainhub@latest \
				lunary@latest loguru@latest devtools@latest  langchain-groq@latest  litellm@latest 

#langchain-openai@latest

clean:  # remove byte code
# find . -type f -name "*.py[co]" -delete -or -type d -name "__pycache__" -delete
	find ./python/ai/__pycache__ -type f -delete
	find ./python/ai/__pycache__ -type d -empty -delete

lint:
	poetry run ruff check --select I --fix
	poetry run ruff format