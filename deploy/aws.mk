# Script to deploy to AWS
# Created by Claude Sonnet-4

#cSpell: disable
AWS_REGION=eu-west-1
AWS_ACCOUNT_ID=909658914353
AWS_SUBNET=subnet-09ea60865f6ded152
AWS_SECURITY_GROUP=sg-0f9ae86b1de2e3956


##############
##  AWS  ###
##############
.PHONY: login_aws push_aws deploy_aws_ecs aws_store_secrets

awd_login:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

aws_push: ## Push to AWS ECR
	aws ecr create-repository --repository-name $(APP) --region $(AWS_REGION) || true
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	docker tag $(APP):$(IMAGE_VERSION) $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION)
	docker push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION)

aws_deploy: ## Deploy to AWS ECS Fargate
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
	
	@echo "Creating IAM role for ECS task..."
	aws iam create-role \
		--role-name ecsTaskRole \
		--assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
		--region $(AWS_REGION) || true
	
	@echo "Creating policy for SSM parameter access..."
	aws iam put-role-policy \
		--role-name ecsTaskRole \
		--policy-name SSMParameterAccess \
		--policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["ssm:GetParameter","ssm:GetParameters"],"Resource":"arn:aws:ssm:$(AWS_REGION):$(AWS_ACCOUNT_ID):parameter/$(APP)/*"}]}' \
		--region $(AWS_REGION) || true
	
	@echo "Waiting for role to be available..."
	sleep 10
	
	@echo "Creating task definition..."
	@chmod +x deploy/generate_container_secrets.sh
	@SECRETS_JSON=$$(./deploy/generate_container_secrets.sh $(APP) $(AWS_REGION) $(AWS_ACCOUNT_ID)); \
	aws ecs register-task-definition \
		--family $(APP)-task \
		--network-mode awsvpc \
		--cpu "256" \
		--memory "512" \
		--requires-compatibilities "FARGATE" \
		--execution-role-arn arn:aws:iam::$(AWS_ACCOUNT_ID):role/ecsTaskExecutionRole \
		--task-role-arn arn:aws:iam::$(AWS_ACCOUNT_ID):role/ecsTaskRole \
		--container-definitions "[{\"name\":\"$(APP)-container\",\"image\":\"$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION)\",\"portMappings\":[{\"containerPort\":443,\"hostPort\":443},{\"containerPort\":8501,\"hostPort\":8501}],\"essential\":true,\"secrets\":$$SECRETS_JSON}]" \
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

aws_get_ecs_url: ## Get the public IP/URL of the deployed ECS service
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
			echo "Your application is available at:"; \
			echo "  HTTPS: https://$$PUBLIC_IP:443 (SSL/TLS enabled)"; \
			echo "  HTTP:  http://$$PUBLIC_IP:8501 (fallback)"; \
			echo ""; \
			echo "ℹ️  Note: HTTPS uses self-signed certificates"; \
			echo "   Your browser will show a security warning - this is normal"; \
		else \
			echo "No public IP found. The task might still be starting."; \
		fi; \
	else \
		echo "No running tasks found. The service might still be starting."; \
		echo "Check service status with: aws ecs describe-services --cluster $(APP)-cluster --services $(APP)-service --region $(AWS_REGION)"; \
	fi

aws_fix_security_group: ## Fix security group to allow port 8501 and 443 access
	@echo "Adding inbound rule for port 8501..."
	@aws ec2 authorize-security-group-ingress \
		--group-id $(AWS_SECURITY_GROUP) \
		--protocol tcp \
		--port 8501 \
		--cidr 0.0.0.0/0 \
		--region $(AWS_REGION) || echo "Rule might already exist"
	@echo "Adding inbound rule for port 443 (HTTPS)..."
	@aws ec2 authorize-security-group-ingress \
		--group-id $(AWS_SECURITY_GROUP) \
		--protocol tcp \
		--port 443 \
		--cidr 0.0.0.0/0 \
		--region $(AWS_REGION) || echo "Rule might already exist"
	@echo "Security group updated!"

aws_debug_ecs: ## Debug ECS deployment issues
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
	@echo "=== Running Tasks Details ==="
	@TASK_ARN=$$(aws ecs list-tasks \
		--cluster $(APP)-cluster \
		--service-name $(APP)-service \
		--region $(AWS_REGION) \
		--query 'taskArns[0]' \
		--output text); \
	if [ "$$TASK_ARN" != "None" ] && [ "$$TASK_ARN" != "" ]; then \
		aws ecs describe-tasks \
			--cluster $(APP)-cluster \
			--tasks $$TASK_ARN \
			--region $(AWS_REGION) \
			--query 'tasks[0].{TaskArn:taskArn,LastStatus:lastStatus,HealthStatus:healthStatus,CreatedAt:createdAt}' \
			--output table; \
	fi
	@echo ""
	@echo "=== Stopped Tasks (if any) ==="
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
		--query 'SecurityGroups[0].IpPermissions' \
		--output table || echo "Security group not found"
	@echo ""
	@echo "=== Port 8501 Specific Rules ==="
	@aws ec2 describe-security-groups \
		--group-ids $(AWS_SECURITY_GROUP) \
		--region $(AWS_REGION) \
		--query 'SecurityGroups[0].IpPermissions[?FromPort==`8501`]' \
		--output table || echo "No port 8501 rules found"
	@echo ""
	@echo "=== Port 443 (HTTPS) Specific Rules ==="
	@aws ec2 describe-security-groups \
		--group-ids $(AWS_SECURITY_GROUP) \
		--region $(AWS_REGION) \
		--query 'SecurityGroups[0].IpPermissions[?FromPort==`443`]' \
		--output table || echo "No port 443 rules found"

aws_store_secrets: ## Store all environment variables from .env file in AWS Systems Manager Parameter Store
	@echo "Storing environment variables from .env file in SSM Parameter Store..."
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Error: .env file not found"; \
		exit 1; \
	fi
	@echo "Reading .env file and uploading to SSM..."
	@grep -v '^#' $(ENV_FILE)| grep -v '^$$' | while IFS='=' read -r key value; do \
		if [ -n "$$key" ] && [ -n "$$value" ]; then \
			echo "Uploading $$key..."; \
			aws ssm put-parameter \
				--name "/$(APP)/$$key" \
				--value "$$value" \
				--type "SecureString" \
				--region $(AWS_REGION) \
				--overwrite || echo "Failed to upload $$key"; \
		fi; \
	done
	@echo "Environment variables stored successfully in SSM Parameter Store"

aws_list_secrets: ## List all stored secrets in AWS SSM Parameter Store
	@echo "Listing stored secrets for $(APP)..."
	@aws ssm describe-parameters \
		--parameter-filters "Key=Name,Option=BeginsWith,Values=/$(APP)/" \
		--region $(AWS_REGION) \
		--query 'Parameters[].{Name:Name,Type:Type,LastModified:LastModifiedDate}' \
		--output table

aws_store_secrets_manual: ## Store specific API keys manually (legacy method)
	@echo "Storing API keys in SSM Parameter Store..."
	@if [ -z "$(OPENROUTER_API_KEY)" ]; then \
		echo "Error: OPENROUTER_API_KEY environment variable is not set"; \
		exit 1; \
	fi
	@if [ -z "$(DEEPSEEK_API_KEY)" ]; then \
		echo "Error: DEEPSEEK_API_KEY environment variable is not set"; \
		exit 1; \
	fi
	aws ssm put-parameter \
		--name "/$(APP)/OPENROUTER_API_KEY" \
		--value "$(OPENROUTER_API_KEY)" \
		--type "SecureString" \
		--region $(AWS_REGION) \
		--overwrite || true
	aws ssm put-parameter \
		--name "/$(APP)/DEEPSEEK_API_KEY" \
		--value "$(DEEPSEEK_API_KEY)" \
		--type "SecureString" \
		--region $(AWS_REGION) \
		--overwrite || true
	@echo "API keys stored successfully in SSM Parameter Store"
