# Makefile

export PROJECT=genai-training
export IMAGE_VERSION=1.2a
export REGISTRY=XXXX.azurecr.io
export STREAMLIT_ENTRY_POINT="python/GenAI_Training.py"

#WARNING : Put the API key into the docker image. NOT RECOMMANDED IN PRODUCTION
build: ## build the image
	docker build --pull --rm -f "Dockerfile" -t $(PROJECT):$(IMAGE_VERSION) "." \
      --build-arg  OPENAI_API=$(OPENAI_API_KEY) \

run: ## execute the image locally
	docker run -it  -p 8000:8000 $(PROJECT):$(IMAGE_VERSION)

check: ## Check if the image is built
	docker images -a

fast_api:  # run Python code localy
	uvicorn python.main_api:app --reload

webapp:
	streamlit run $(STREAMLIT_ENTRY_POINT)


sync_time:  # Needed because WSL loose time after hibernation, and that can cause issues when pushing 
	sudo hwclock -s 

test:
	pytest -s
	
push:  # Push to a registry
	docker tag $(PROJECT):$(IMAGE_VERSION) $(REGISTRY)/$(PROJECT):$(IMAGE_VERSION)
	docker push $(REGISTRY)/$(PROJECT):$(IMAGE_VERSION)

save:  # Create a zipped version of the image
	docker save $(PROJECT):$(IMAGE_VERSION)| gzip > /tmp/$(PROJECT)_$(IMAGE_VERSION).tar.gz

update:  # Update selected fast changing dependencies
	poetry add 	langchain@latest langchain-experimental@latest  langchain-core@latest  langchain-community@latest langgraph@latest langchainhub@latest \
				lunary@latest loguru@latest devtools@latest  langchain-groq@latest  

#langchain-openai@latest

clean:  # remove byte code
	find . -type f -name "*.py[co]" -delete -or -type d -name "__pycache__" -delete

