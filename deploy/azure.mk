
#cSpell: disable
APP=genai-blueprint

MODAL_ENTRY_POINT="src/main/modal_app.py"
IMAGE_VERSION=0.2a
REGISTRY_AZ=XXXX.azurecr.io
REGISTRY_NAME=XXX
LOCATION=europe-west4
PROJECT_ID_GCP=XXX


AWS_REGION=eu-west-1
AWS_ACCOUNT_ID=909658914353
AWS_SUBNET=subnet-09ea60865f6ded152
AWS_SECURITY_GROUP=sg-0f9ae86b1de2e3956


##############################
##  Build Docker, and run locally
##############################
#WARNING : Put the API key into the docker image. NOT RECOMMANDED IN PRODUCTION

test1:
	echo $(OPENAI_API_KEY)
	echo $(ENV_FILE)

.PHONY: build run save sync_time check # Docker build and deployment 


check: ## Check if the image is built
	docker images -a

sync-time: # Needed because WSL loose time after hibernation, and that can cause issues when pushing
	sudo hwclock -s 

build: ## Build the docker image
	docker build --pull --rm -f "Dockerfile" -t $(APP):$(IMAGE_VERSION) "."

run: ## Execute the image with environment variables
	docker run -it -p 8000:8000 -p 8501:8501 \
		-e OPENROUTER_API_KEY=$(OPENROUTER_API_KEY) \
		-e DEEPSEEK_API_KEY=$(DEEPSEEK_API_KEY) \
		$(APP):$(IMAGE_VERSION)

save:  # Create a zipped version of the image
	docker save $(APP):$(IMAGE_VERSION)| gzip > /tmp/$(APP)_$(IMAGE_VERSION).tar.gz


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
		
##############
##  AZURE  ###
##############
.PHONY: push_az # Azure targets
	
push-az: ## Push to a Azure registry
	docker tag $(APP):$(IMAGE_VERSION) $(REGISTRY_AZ)/$(APP):$(IMAGE_VERSION)
	docker push $(REGISTRY_AZ)/$(APP):$(IMAGE_VERSION)

# To be completed...


##############
##  AWS  ###
##############
.PHONY: login_aws push_aws deploy_aws_ecs

login-aws:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

push-aws: ## Push to AWS ECR
	aws ecr create-repository --repository-name $(APP) --region $(AWS_REGION) || true
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	docker tag $(APP):$(IMAGE_VERSION) $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION)
	docker push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION)

deploy-aws-ecs: ## Deploy to AWS ECS Fargate
	@echo "Creating ECS cluster..."
	aws ecs create-cluster --cluster-name $(APP)-cluster --region $(AWS_REGION) || true
	
	@echo "Creating IAM role for ECS task execution..."
	aws iam create-role \
		--role-name ecsTaskExecutionRole \
		--assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
		--region $(AWS_REGION) || true
	
	@echo "Attaching policy to execution role..."
	aws iam attach-role-policy \
		--role-name ecsTaskExecutionRole \
		--policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
		--region $(AWS_REGION) || true
	
	@echo "Waiting for role to be available..."
	sleep 10
	
	@echo "Creating task definition..."
	aws ecs register-task-definition \
		--family $(APP)-task \
		--network-mode awsvpc \
		--cpu "256" \
		--memory "512" \
		--requires-compatibilities "FARGATE" \
		--execution-role-arn arn:aws:iam::$(AWS_ACCOUNT_ID):role/ecsTaskExecutionRole \
		--container-definitions '[{"name":"$(APP)-container","image":"$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION)","portMappings":[{"containerPort":8501,"hostPort":8501}],"essential":true}]' \
		--region $(AWS_REGION)
	
	@echo "Creating or updating ECS service..."
	aws ecs create-service \
		--cluster $(APP)-cluster \
		--service-name $(APP)-service \
		--task-definition $(APP)-task \
		--desired-count 1 \
		--launch-type "FARGATE" \
		--network-configuration "awsvpcConfiguration={subnets=[$(AWS_SUBNET)],securityGroups=[$(AWS_SECURITY_GROUP)],assignPublicIp=ENABLED}" \
		--region $(AWS_REGION) || \
	aws ecs update-service \
		--cluster $(APP)-cluster \
		--service $(APP)-service \
		--task-definition $(APP)-task \
		--desired-count 1 \
		--region $(AWS_REGION)
	
	@echo "Application deployed! It may take a few minutes to become available."

get-aws-ecs-url: ## Get the public IP/URL of the deployed ECS service
	@echo "Getting ECS service status..."
	@aws ecs describe-services \
		--cluster $(APP)-cluster \
		--services $(APP)-service \
		--region $(AWS_REGION) \
		--query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount}' \
		--output table
	@echo "Getting running tasks..."
	@TASK_ARN=$$(aws ecs list-tasks \
		--cluster $(APP)-cluster \
		--service-name $(APP)-service \
		--region $(AWS_REGION) \
		--query 'taskArns[0]' \
		--output text); \
	if [ "$$TASK_ARN" != "None" ] && [ "$$TASK_ARN" != "" ]; then \
		echo "Found task: $$TASK_ARN"; \
		PUBLIC_IP=$$(aws ecs describe-tasks \
			--cluster $(APP)-cluster \
			--tasks $$TASK_ARN \
			--region $(AWS_REGION) \
			--query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
			--output text | xargs -I {} aws ec2 describe-network-interfaces \
			--network-interface-ids {} \
			--region $(AWS_REGION) \
			--query 'NetworkInterfaces[0].Association.PublicIp' \
			--output text); \
		if [ "$$PUBLIC_IP" != "None" ] && [ "$$PUBLIC_IP" != "" ]; then \
			echo "Your application is available at: http://$$PUBLIC_IP:8501"; \
		else \
			echo "No public IP found. The task might still be starting."; \
		fi; \
	else \
		echo "No running tasks found. The service might still be starting."; \
		echo "Check service status with: aws ecs describe-services --cluster $(APP)-cluster --services $(APP)-service --region $(AWS_REGION)"; \
	fi

debug-aws-ecs: ## Debug ECS deployment issues
	@echo "=== ECS Service Events ==="
	@aws ecs describe-services \
		--cluster $(APP)-cluster \
		--services $(APP)-service \
		--region $(AWS_REGION) \
		--query 'services[0].events[0:5]' \
		--output table
	@echo ""
	@echo "=== Task Definition Details ==="
	@aws ecs describe-task-definition \
		--task-definition $(APP)-task \
		--region $(AWS_REGION) \
		--query 'taskDefinition.{Family:family,Revision:revision,Status:status,Cpu:cpu,Memory:memory,ExecutionRoleArn:executionRoleArn}' \
		--output table
	@echo ""
	@echo "=== All Tasks (including stopped) ==="
	@aws ecs list-tasks \
		--cluster $(APP)-cluster \
		--service-name $(APP)-service \
		--region $(AWS_REGION) \
		--desired-status STOPPED \
		--query 'taskArns' \
		--output text | head -1 | xargs -I {} aws ecs describe-tasks \
		--cluster $(APP)-cluster \
		--tasks {} \
		--region $(AWS_REGION) \
		--query 'tasks[0].{TaskArn:taskArn,LastStatus:lastStatus,StoppedReason:stoppedReason,StoppedAt:stoppedAt}' \
		--output table 2>/dev/null || echo "No stopped tasks found"
	@echo ""
	@echo "=== Security Group Check ==="
	@aws ec2 describe-security-groups \
		--group-ids $(AWS_SECURITY_GROUP) \
		--region $(AWS_REGION) \
		--query 'SecurityGroups[0].IpPermissions[?FromPort==`8501`]' \
		--output table || echo "Security group not found or no port 8501 rule"

##############
##  MODAL  ###
##############

.PHONY: modal_install modal_login modal_deploy modal_deploy_force modal_run modal_secrets modal_clear_cache

modal-install: ## Install Modal CLI
	uv pip install modal

modal-login: ## Login to Modal
	modal token new

modal-deploy: ## Deploy to Modal
	modal deploy $(MODAL_ENTRY_POINT)

modal-serve:
	modal serve $(MODAL_ENTRY_POINT)

modal-deploy-force: ## Deploy to Modal with forced image rebuild
	modal image clear
	modal deploy $(MODAL_ENTRY_POINT)

modal-run: ## Run locally with Modal
	modal run $(MODAL_ENTRY_POINT)

modal-clear-cache: ## Clear Modal image cache
	modal image clear

modal-secrets: ## Create Modal secrets from .env file
	modal secret create genai-secrets $$(cat .env | xargs)
