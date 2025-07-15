AWS_REGION=eu-west-1
AWS_ACCOUNT_ID=909658914353
AWS_SUBNET=subnet-09ea60865f6ded152
AWS_SECURITY_GROUP=sg-0f9ae86b1de2e3956

.PHONY: aws_push aws_deploy aws_shell aws_logs

aws_push: ## Push Docker image to AWS ECR
	aws ecr create-repository --repository-name $(APP) --region $(AWS_REGION) || true
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	docker tag $(APP):$(IMAGE_VERSION) $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION)
	docker push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION)

aws_deploy: ## Deploy to AWS ECS Fargate
	@echo "Creating ECS cluster..."
	aws ecs create-cluster --cluster-name $(APP)-cluster --region $(AWS_REGION) || true
	
	@echo "Creating task definition..."
	@SECRETS_JSON=$$(./deploy/generate_container_secrets.sh $(APP) $(AWS_REGION) $(AWS_ACCOUNT_ID) $(ENV_FILE)); \
	aws ecs register-task-definition \
		--family $(APP)-task \
		--network-mode awsvpc \
		--cpu "256" \
		--memory "512" \
		--requires-compatibilities "FARGATE" \
		--execution-role-arn arn:aws:iam::$(AWS_ACCOUNT_ID):role/ecsTaskExecutionRole \
		--task-role-arn arn:aws:iam::$(AWS_ACCOUNT_ID):role/ecsTaskRole \
		--container-definitions "[{\"name\":\"$(APP)-container\",\"image\":\"$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION)\",\"portMappings\":[{\"containerPort\":8501,\"hostPort\":8501}],\"essential\":true,\"secrets\":$$SECRETS_JSON,\"logConfiguration\":{\"logDriver\":\"awslogs\",\"options\":{\"awslogs-group\":\"/ecs/$(APP)-task\",\"awslogs-region\":\"$(AWS_REGION)\",\"awslogs-stream-prefix\":\"ecs\"}}}]" \
		--region $(AWS_REGION)

	@echo "Creating or updating ECS service..."
	aws ecs create-service \
		--cluster $(APP)-cluster \
		--service-name $(APP)-service \
		--task-definition $(APP)-task \
		--desired-count 1 \
		--launch-type "FARGATE" \
		--enable-execute-command \
		--network-configuration "awsvpcConfiguration={subnets=[$(AWS_SUBNET)],securityGroups=[$(AWS_SECURITY_GROUP)],assignPublicIp=ENABLED}" \
		--region $(AWS_REGION) || \
	aws ecs update-service \
		--cluster $(APP)-cluster \
		--service $(APP)-service \
		--task-definition $(APP)-task \
		--desired-count 1 \
		--enable-execute-command \
		--region $(AWS_REGION)
	
	@echo "Application deployed! Access it at the public IP on port 8501."

aws_shell: ## Open a shell in the running ECS container
	@TASK_ARN=$$(aws ecs list-tasks \
		--cluster $(APP)-cluster \
		--service-name $(APP)-service \
		--region $(AWS_REGION) \
		--query 'taskArns[0]' \
		--output text); \
	if [ "$$TASK_ARN" != "None" ]; then \
		aws ecs execute-command \
			--cluster $(APP)-cluster \
			--task $$TASK_ARN \
			--container $(APP)-container \
			--interactive \
			--command "/bin/bash" \
			--region $(AWS_REGION); \
	else \
		echo "No running tasks found"; \
	fi

aws_logs: ## Get application logs from CloudWatch
	@TASK_ARN=$$(aws ecs list-tasks \
		--cluster $(APP)-cluster \
		--service-name $(APP)-service \
		--region $(AWS_REGION) \
		--query 'taskArns[0]' \
		--output text); \
	if [ "$$TASK_ARN" != "None" ]; then \
		TASK_ID=$$(echo $$TASK_ARN | cut -d'/' -f3); \
		aws logs get-log-events \
			--log-group-name "/ecs/$(APP)-task" \
			--log-stream-name "ecs/$(APP)-container/$$TASK_ID" \
			--start-time $$(date -d '30 minutes ago' +%s)000 \
			--region $(AWS_REGION) \
			--query 'events[].message' \
			--output text 2>/dev/null || echo "No logs found"; \
	else \
		echo "No running tasks found"; \
	fi


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
			echo "  HTTP: http://$$PUBLIC_IP:8501"; \
			echo ""; \
			echo "🔍 Troubleshooting:"; \
		else \
			echo "No public IP found. The task might still be starting."; \
		fi; \
	else \
		echo "No running tasks found. The service might still be starting."; \
		echo "Check service status with: aws ecs describe-services --cluster $(APP)-cluster --services $(APP)-service --region $(AWS_REGION)"; \
	fi