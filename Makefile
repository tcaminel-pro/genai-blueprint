# Makefile

# General setttings, notably for deployement
export APP=genai-tcl
export IMAGE_VERSION=0.2a
export REGISTRY_AZ=XXXX.azurecr.io

export REGISTRY_NAME=XXX
export LOCATION=europe-west4
export PROJECT_ID_GCP=XXX

export STREAMLIT_ENTRY_POINT="python/GenAI_Lab.py"

topdir := $(shell pwd)

#WARNING : Put the API key into the docker image. NOT RECOMMANDED IN PRODUCTION


check: ## Check if the image is built
	docker images -a

fast_api:  # run Python code localy
	uvicorn python.fastapi_app:app --reload

langserve:
	python python/langserve_app.py

webapp:
	streamlit run $(STREAMLIT_ENTRY_POINT)


sync_time:  # Needed because WSL loose time after hibernation, and that can cause issues when pushing 
	sudo hwclock -s 

test:
	pytest -s

rebase:
	git fetch origin
	git stash
	git rebase origin/main

######################
##  Build Docker, and run locally
#####################
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


update:  # Update selected fast changing dependencies 
	poetry add 	langchain@latest  langchain-core@latest langgraph@latest langserve@latest langchainhub@latest \
				 langchain-groq@latest  \
				 langchain-experimental@latest   langchain-community@latest
# litellm@latest lunary@latest

#langchain-openai@latest

clean:  # remove byte code  DOES NOT WORK
# find . -type f -name "*.py[co]" -delete -or -type d -name "__pycache__" -delete
	find ./python/ai/__pycache__ -type f -delete
	find ./python/ai/__pycache__ -type d -empty -delete

lint:
	poetry run ruff check --select I --fix
	poetry run ruff format

backup:
	cp ~/.keys.sh ~/ln_to_onedrive/backup/wsl/tcl/
	cp ~/.bashrc ~/ln_to_onedrive/backup/wsl/tcl/
	cp ~/.dev.bash-profile ~/ln_to_onedrive/backup/wsl/tcl/