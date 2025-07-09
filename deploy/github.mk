
#cSpell: disable

##############################################
##  GitHub Container Registry Deployment   ##
##############################################

# GitHub Container Registry configuration
GITHUB_REGISTRY=ghcr.io
GITHUB_USERNAME ?= $(shell git config user.name | tr '[:upper:]' '[:lower:]')
GITHUB_REPO ?= $(shell basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]')
GITHUB_IMAGE_NAME=$(GITHUB_REGISTRY)/$(GITHUB_USERNAME)/$(GITHUB_REPO)

# Image versioning
GIT_COMMIT=$(shell git rev-parse --short HEAD)
GIT_BRANCH=$(shell git rev-parse --abbrev-ref HEAD | sed 's/[^a-zA-Z0-9._-]/-/g')
TIMESTAMP=$(shell date +%Y%m%d-%H%M%S)

# Default tags
GITHUB_TAG_LATEST=$(GITHUB_IMAGE_NAME):latest
GITHUB_TAG_COMMIT=$(GITHUB_IMAGE_NAME):$(GIT_COMMIT)
GITHUB_TAG_BRANCH=$(GITHUB_IMAGE_NAME):$(GIT_BRANCH)
GITHUB_TAG_VERSION=$(GITHUB_IMAGE_NAME):$(IMAGE_VERSION)
GITHUB_TAG_TIMESTAMP=$(GITHUB_IMAGE_NAME):$(TIMESTAMP)

.PHONY: github_login github_build github_tag github_push github_deploy github_info github_clean

github_info: ## Show GitHub registry information
    @echo "GitHub Registry Info:"
    @echo "  Registry: $(GITHUB_REGISTRY)"
    @echo "  Username: $(GITHUB_USERNAME)"
    @echo "  Repository: $(GITHUB_REPO)"
    @echo "  Image Name: $(GITHUB_IMAGE_NAME)"
    @echo "  Git Commit: $(GIT_COMMIT)"
    @echo "  Git Branch: $(GIT_BRANCH)"
    @echo "  Version: $(IMAGE_VERSION)"
    @echo "Available tags:"
    @echo "  Latest: $(GITHUB_TAG_LATEST)"
    @echo "  Commit: $(GITHUB_TAG_COMMIT)"
    @echo "  Branch: $(GITHUB_TAG_BRANCH)"
    @echo "  Version: $(GITHUB_TAG_VERSION)"
    @echo "  Timestamp: $(GITHUB_TAG_TIMESTAMP)"

github_login: ## Login to GitHub Container Registry
    @echo "Logging into GitHub Container Registry..."
    @echo "Make sure you have a GitHub Personal Access Token with 'write:packages' scope"
    @echo "Create token at: https://github.com/settings/tokens"
    docker login $(GITHUB_REGISTRY) -u $(GITHUB_USERNAME)

github_build: ## Build Docker image for GitHub registry
    @echo "Building Docker image for GitHub registry..."
    docker build --pull --rm -f "deploy/Dockerfile" -t $(GITHUB_TAG_LATEST) "."

github_tag: ## Tag the local image with multiple GitHub registry tags
    @echo "Tagging image with multiple tags..."
    docker tag $(APP):$(IMAGE_VERSION) $(GITHUB_TAG_LATEST)
    docker tag $(APP):$(IMAGE_VERSION) $(GITHUB_TAG_COMMIT)
    docker tag $(APP):$(IMAGE_VERSION) $(GITHUB_TAG_BRANCH)
    docker tag $(APP):$(IMAGE_VERSION) $(GITHUB_TAG_VERSION)
    docker tag $(APP):$(IMAGE_VERSION) $(GITHUB_TAG_TIMESTAMP)
    @echo "Tagged with:"
    @echo "  $(GITHUB_TAG_LATEST)"
    @echo "  $(GITHUB_TAG_COMMIT)"
    @echo "  $(GITHUB_TAG_BRANCH)"
    @echo "  $(GITHUB_TAG_VERSION)"
    @echo "  $(GITHUB_TAG_TIMESTAMP)"

github_push: ## Push all tagged images to GitHub registry
    @echo "Pushing images to GitHub Container Registry..."
    docker push $(GITHUB_TAG_LATEST)
    docker push $(GITHUB_TAG_COMMIT)
    docker push $(GITHUB_TAG_BRANCH)
    docker push $(GITHUB_TAG_VERSION)
    docker push $(GITHUB_TAG_TIMESTAMP)
    @echo "Successfully pushed all tags to GitHub Container Registry"

github_deploy: docker_build github_tag github_push ## Complete deployment: build, tag, and push to GitHub registry
    @echo "GitHub Container Registry deployment complete!"
    @echo "Image available at: $(GITHUB_TAG_LATEST)"
    @echo ""
    @echo "To use this image with Modal:"
    @echo "  MODAL_AWS_IMAGE_URI=$(GITHUB_TAG_LATEST) make modal_deploy_aws"

github_deploy_direct: ## Build and deploy directly to GitHub registry (without local tagging)
    @echo "Building and pushing directly to GitHub Container Registry..."
    docker build --pull --rm -f "deploy/Dockerfile" \
        -t $(GITHUB_TAG_LATEST) \
        -t $(GITHUB_TAG_COMMIT) \
        -t $(GITHUB_TAG_BRANCH) \
        -t $(GITHUB_TAG_VERSION) \
        -t $(GITHUB_TAG_TIMESTAMP) \
        "."
    docker push $(GITHUB_TAG_LATEST)
    docker push $(GITHUB_TAG_COMMIT)
    docker push $(GITHUB_TAG_BRANCH)
    docker push $(GITHUB_TAG_VERSION)
    docker push $(GITHUB_TAG_TIMESTAMP)
    @echo "Direct deployment complete!"

github_clean: ## Remove local GitHub registry tagged images
    @echo "Cleaning up local GitHub registry tags..."
    -docker rmi $(GITHUB_TAG_LATEST)
    -docker rmi $(GITHUB_TAG_COMMIT)
    -docker rmi $(GITHUB_TAG_BRANCH)
    -docker rmi $(GITHUB_TAG_VERSION)
    -docker rmi $(GITHUB_TAG_TIMESTAMP)
    @echo "Cleanup complete"

github_pull: ## Pull the latest image from GitHub registry
    docker pull $(GITHUB_TAG_LATEST)

github_run: ## Run the GitHub registry image locally
    docker run -it -p 8000:8000 -p 8501:8501 \
        -e OPENROUTER_API_KEY=$(OPENROUTER_API_KEY) \
        -e DEEPSEEK_API_KEY=$(DEEPSEEK_API_KEY) \
        $(GITHUB_TAG_LATEST)