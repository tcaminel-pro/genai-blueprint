
#cSpell: disable
REGISTRY_NAME=XXX
LOCATION=europe-west4
PROJECT_ID_GCP=XXX

##############
##  GCP  ###
##############

.PHONY: login_gcp build_gcp push_gcp create_repo_gcp # GCP targets

# To be completed...

login-gcp:
	gcloud auth login
	gcloud config set project  $(PROJECT_ID_GCP)

build-gcp: ## build the image gor GCP
	docker build -t gcr.io/$(PROJECT_ID_GCP)/$(APP):$(IMAGE_VERSION) . --build-arg OPENAI_API=$(OPENAI_API_KEY) 

push-gcp: ## Push to a GCP registry
# gcloud auth configure-docker
	docker tag $(APP):$(IMAGE_VERSION) $(LOCATION)-docker.pkg.dev/$(PROJECT_ID_GCP)/$(REGISTRY_NAME)/$(APP):$(IMAGE_VERSION)
	docker push $(LOCATION)-docker.pkg.dev/$(PROJECT_ID_GCP)/$(REGISTRY_NAME)/$(APP):$(IMAGE_VERSION)
# gcloud run deploy --image gcr.io/$(PROJECT_ID_GCP)/$(APP):$(IMAGE_VERSION) --platform managed

create-repo-gcp:
	gcloud auth configure-docker $(LOCATION)-docker.pkg.dev
	gcloud artifacts repositories create $(REGISTRY_NAME) --repository-format=docker \
		--location=$(LOCATION) --description="Docker repository" \
		--project=$(PROJECT_ID_GCP)
	
