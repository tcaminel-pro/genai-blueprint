AWS_REGION=eu-west-1
AWS_SUBNET=subnet-09ea60865f6ded152
AWS_SECURITY_GROUP=sg-0f9ae86b1de2e3956

.PHONY: aws_push aws_deploy aws_shell aws_logs aws_debug aws_fix_security_group aws_test_connection aws_check_container aws_redeploy

aws-push: ## Push Docker image to AWS ECR
	aws ecr create-repository --repository-name $(APP) --region $(AWS_REGION) || true
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	docker tag $(APP):$(IMAGE_VERSION) $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION)
	docker push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION)

aws-deploy: ## Deploy to AWS ECS Fargate
	@echo "Creating ECS cluster..."
	aws --no-cli-pager ecs create-cluster --cluster-name $(APP)-cluster --region $(AWS_REGION) || true
	
	@echo "Creating CloudWatch log group..."
	aws --no-cli-pager logs create-log-group \
		--log-group-name "/ecs/$(APP)-task" \
		--region $(AWS_REGION) || true
	
	@echo "Updating security group to allow HTTP traffic..."
	aws --no-cli-pager ec2 authorize-security-group-ingress \
		--group-id $(AWS_SECURITY_GROUP) \
		--ip-permissions '[{"IpProtocol": "tcp", "FromPort": 8501, "ToPort": 8501, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}]' \
		--region $(AWS_REGION) || true
	
	@echo "Creating task definition..."
	@SECRETS_JSON=$$(./deploy/generate_container_secrets.sh $(APP) $(AWS_REGION) $(AWS_ACCOUNT_ID) $(ENV_FILE)); \
	aws --no-cli-pager ecs register-task-definition \
		--family $(APP)-task \
		--network-mode awsvpc \
		--cpu "256" \
		--memory "512" \
		--requires-compatibilities "FARGATE" \
		--execution-role-arn arn:aws:iam::$(AWS_ACCOUNT_ID):role/ecsTaskExecutionRole \
		--task-role-arn arn:aws:iam::$(AWS_ACCOUNT_ID):role/ecsTaskRole \
		--container-definitions "[{\"name\":\"$(APP)-container\",\"image\":\"$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP):$(IMAGE_VERSION)\",\"portMappings\":[{\"containerPort\":8501,\"hostPort\":8501}],\"essential\":true,\"secrets\":$$SECRETS_JSON,\"healthCheck\":{\"command\":[\"CMD-SHELL\",\"curl -f http://localhost:8501/_stcore/health || exit 1\"],\"interval\":30,\"timeout\":5,\"retries\":3},\"logConfiguration\":{\"logDriver\":\"awslogs\",\"options\":{\"awslogs-group\":\"/ecs/$(APP)-task\",\"awslogs-region\":\"$(AWS_REGION)\",\"awslogs-stream-prefix\":\"ecs\"}}}]" \
		--region $(AWS_REGION)

	@echo "Creating or updating ECS service..."
	aws --no-cli-pager ecs create-service \
		--cluster $(APP)-cluster \
		--service-name $(APP)-service \
		--task-definition $(APP)-task \
		--desired-count 1 \
		--launch-type "FARGATE" \
		--enable-execute-command \
		--network-configuration "awsvpcConfiguration={subnets=[$(AWS_SUBNET)],securityGroups=[$(AWS_SECURITY_GROUP)],assignPublicIp=ENABLED}" \
		--region $(AWS_REGION) || \
	aws --no-cli-pager ecs update-service \
		--cluster $(APP)-cluster \
		--service $(APP)-service \
		--task-definition $(APP)-task \
		--desired-count 1 \
		--enable-execute-command \
		--region $(AWS_REGION)
	
	@echo "Application deployed! Access it at the public IP on port 8501."

aws-shell: ## Open a shell in the running ECS container
	@TASK_ARN=$$(aws ecs list-tasks \
		--cluster $(APP)-cluster \
		--service-name $(APP)-service \
		--region $(AWS_REGION) \
		--query 'taskArns[0]' \
		--output text); \
	if [ "$$TASK_ARN" != "None" ]; then \
		aws --no-cli-pager ecs execute-command \
			--cluster $(APP)-cluster \
			--task $$TASK_ARN \
			--container $(APP)-container \
			--interactive \
			--command "/bin/bash" \
			--region $(AWS_REGION); \
	else \
		echo "No running tasks found"; \
	fi

aws-redeploy: ## Force a new deployment and wait for it to stabilize
	@if [ -z "$(APP)" ]; then \
		echo "Error: APP variable not set. Usage: make aws_redeploy APP=<your-app-name>"; \
		exit 1; \
	fi
	@echo "=== Forcing new deployment ==="
	aws --no-cli-pager ecs update-service \
		--cluster $(APP)-cluster \
		--service $(APP)-service \
		--force-new-deployment \
		--region $(AWS_REGION)
	@echo "Waiting for deployment to stabilize (timeout: 10 minutes)..."
	timeout 600 aws ecs wait services-stable \
		--cluster $(APP)-cluster \
		--services $(APP)-service \
		--region $(AWS_REGION) || echo "⚠️  Warning: Service stabilization timed out - check manually"
	@echo "✅ Deployment completed"
	@echo ""
	@echo "Checking deployment logs..."
	@$(MAKE) aws_logs APP=$(APP)
	@echo ""
	@$(MAKE) aws_get_ecs_url APP=$(APP)

aws-check-container: ## Check container configuration and Dockerfile
	@echo "=== Checking Container Configuration ==="
	@echo "Current task definition:"
	@aws ecs describe-task-definition \
		--task-definition $(APP)-task \
		--region $(AWS_REGION) \
		--query 'taskDefinition.containerDefinitions[0].{Image:image,Command:command,EntryPoint:entryPoint,PortMappings:portMappings}' \
		--output table
	@echo ""
	@echo "=== Dockerfile Check ==="
	@if [ -f "deploy/Dockerfile" ]; then \
		echo "Dockerfile exists at deploy/Dockerfile. Checking EXPOSE and CMD:"; \
		grep -n "EXPOSE\|CMD\|ENTRYPOINT" deploy/Dockerfile || echo "No EXPOSE/CMD/ENTRYPOINT found"; \
	elif [ -f "Dockerfile" ]; then \
		echo "Dockerfile exists in root. Checking EXPOSE and CMD:"; \
		grep -n "EXPOSE\|CMD\|ENTRYPOINT" Dockerfile || echo "No EXPOSE/CMD/ENTRYPOINT found"; \
	else \
		echo "❌ No Dockerfile found in current directory or deploy/ directory"; \
	fi
	@echo ""
	@echo "=== Streamlit Configuration Suggestions ==="
	@echo "For Streamlit to work in ECS, ensure your Dockerfile has:"
	@echo "  EXPOSE 8501"
	@echo "  CMD [\"streamlit\", \"run\", \"your_app.py\", \"--server.address=0.0.0.0\", \"--server.port=8501\"]"

aws-test-connection: ## Test connection to the deployed application
	@echo "=== Testing Connection to ECS Application ==="
	@TASK_ARN=$$(aws ecs list-tasks \
		--cluster $(APP)-cluster \
		--service-name $(APP)-service \
		--region $(AWS_REGION) \
		--query 'taskArns[0]' \
		--output text); \
	if [ "$$TASK_ARN" != "None" ] && [ "$$TASK_ARN" != "" ]; then \
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
			echo "Testing connection to: http://$$PUBLIC_IP:8501"; \
			echo ""; \
			echo "=== Basic connectivity test ==="; \
			timeout 5 nc -zv $$PUBLIC_IP 8501 2>&1 || echo "Port 8501 is not reachable"; \
			echo ""; \
			echo "=== HTTP response test ==="; \
			timeout 10 curl -v "http://$$PUBLIC_IP:8501" 2>&1 || echo "HTTP request failed"; \
			echo ""; \
			echo "=== Testing health endpoint ==="; \
			timeout 10 curl -v "http://$$PUBLIC_IP:8501/_stcore/health" 2>&1 || echo "Health endpoint failed"; \
		else \
			echo "No public IP found"; \
		fi; \
	else \
		echo "No running tasks found"; \
	fi

aws-logs: ## Get application logs from CloudWatch
	@TASK_ARN=$$(aws ecs list-tasks \
		--cluster $(APP)-cluster \
		--service-name $(APP)-service \
		--region $(AWS_REGION) \
		--query 'taskArns[0]' \
		--output text); \
	if [ "$$TASK_ARN" != "None" ]; then \
		TASK_ID=$$(echo $$TASK_ARN | cut -d'/' -f3); \
		echo "Getting logs for task: $$TASK_ID"; \
		echo "Log stream: ecs/$(APP)-container/$$TASK_ID"; \
		aws logs get-log-events \
			--log-group-name "/ecs/$(APP)-task" \
			--log-stream-name "ecs/$(APP)-container/$$TASK_ID" \
			--start-time $$(date -d '2 hours ago' +%s)000 \
			--region $(AWS_REGION) \
			--query 'events[].message' \
			--output text 2>/dev/null || echo "No logs found for current task"; \
		echo ""; \
		echo "=== Checking all recent log streams ==="; \
		for stream in $$(aws logs describe-log-streams \
			--log-group-name "/ecs/$(APP)-task" \
			--region $(AWS_REGION) \
			--order-by LastEventTime \
			--descending \
			--max-items 3 \
			--query 'logStreams[].logStreamName' \
			--output text 2>/dev/null); do \
			echo "--- Stream: $$stream ---"; \
			aws logs get-log-events \
				--log-group-name "/ecs/$(APP)-task" \
				--log-stream-name "$$stream" \
				--start-time $$(date -d '2 hours ago' +%s)000 \
				--region $(AWS_REGION) \
				--query 'events[-10:].message' \
				--output text 2>/dev/null || echo "No logs in this stream"; \
		done; \
	else \
		echo "No running tasks found"; \
	fi


aws-get-ecs-url: ## Get the public IP/URL of the deployed ECS service
	@echo "Getting ECS service status..."
	@aws --no-cli-pager ecs describe-services \
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
			echo "🔍 Testing connectivity..."; \
			timeout 10 curl -s -o /dev/null -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" "http://$$PUBLIC_IP:8501" || echo "❌ Connection failed - see troubleshooting below"; \
			echo ""; \
			echo "🔍 Troubleshooting:"; \
			echo "  1. Check logs: make aws_logs"; \
			echo "  2. Check task health: make aws_debug"; \
			echo "  3. Verify security group allows port 8501 from your IP"; \
		else \
			echo "No public IP found. The task might still be starting."; \
		fi; \
	else \
		echo "No running tasks found. The service might still be starting."; \
		echo "Check service status with: aws ecs describe-services --cluster $(APP)-cluster --services $(APP)-service --region $(AWS_REGION)"; \
	fi

aws-fix-security-group: ## Fix security group rules for port 8501
	@echo "Checking and fixing security group rules..."
	@echo "Current security group rules:"
	@aws --no-cli-pager ec2 describe-security-groups \
		--group-ids $(AWS_SECURITY_GROUP) \
		--region $(AWS_REGION) \
		--query 'SecurityGroups[0].{Inbound:IpPermissions,Outbound:IpPermissionsEgress}' \
		--output table || true
	@echo "Adding rule for port 8501 if not exists..."
	@aws --no-cli-pager ec2 authorize-security-group-ingress \
		--group-id $(AWS_SECURITY_GROUP) \
		--ip-permissions '[{"IpProtocol": "tcp", "FromPort": 8501, "ToPort": 8501, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}]' \
		--region $(AWS_REGION) 2>/dev/null && echo "✅ Inbound rule added successfully" || echo "ℹ️  Inbound rule already exists or failed to add"
	@aws --no-cli-pager ec2 authorize-security-group-egress \
		--group-id $(AWS_SECURITY_GROUP) \
		--ip-permissions '[{"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}]' \
		--region $(AWS_REGION) 2>/dev/null && echo "✅ Outbound rule added successfully" || echo "ℹ️  Outbound rule already exists or failed to add"
	@echo "Updated security group rules:"
	@aws --no-cli-pager ec2 describe-security-groups \
		--group-ids $(AWS_SECURITY_GROUP) \
		--region $(AWS_REGION) \
		--query 'SecurityGroups[0].IpPermissions[?FromPort==`8501`]' \
		--output table

aws-debug: ## Debug ECS deployment issues
	@echo "=== ECS Task Health Check ==="
	@TASK_ARN=$$(aws ecs list-tasks \
		--cluster $(APP)-cluster \
		--service-name $(APP)-service \
		--region $(AWS_REGION) \
		--query 'taskArns[0]' \
		--output text); \
	if [ "$$TASK_ARN" != "None" ] && [ "$$TASK_ARN" != "" ]; then \
		echo "Task ARN: $$TASK_ARN"; \
		echo ""; \
		echo "=== Task Status ==="; \
		aws ecs --no-cli-pager describe-tasks \
			--cluster $(APP)-cluster \
			--tasks $$TASK_ARN \
			--region $(AWS_REGION) \
			--query 'tasks[0].{LastStatus:lastStatus,HealthStatus:healthStatus,CreatedAt:createdAt,StartedAt:startedAt}' \
			--output table; \
		echo ""; \
		echo "=== Container Status ==="; \
		aws ecs --no-cli-pager describe-tasks \
			--cluster $(APP)-cluster \
			--tasks $$TASK_ARN \
			--region $(AWS_REGION) \
			--query 'tasks[0].containers[0].{Name:name,LastStatus:lastStatus,HealthStatus:healthStatus,ExitCode:exitCode,Reason:reason}' \
			--output table; \
		echo ""; \
		echo "=== Network Configuration ==="; \
		ENI_ID=$$(aws ecs describe-tasks \
			--cluster $(APP)-cluster \
			--tasks $$TASK_ARN \
			--region $(AWS_REGION) \
			--query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
			--output text); \
		if [ "$$ENI_ID" != "" ]; then \
			echo "Network Interface: $$ENI_ID"; \
			aws --no-cli-pager ec2 describe-network-interfaces \
				--network-interface-ids $$ENI_ID \
				--region $(AWS_REGION) \
				--query 'NetworkInterfaces[0].{PublicIp:Association.PublicIp,PrivateIp:PrivateIpAddress,SecurityGroups:Groups[].GroupId}' \
				--output table; \
		fi; \
		echo ""; \
		echo "=== Recent Logs (last 20 lines) ==="; \
		TASK_ID=$$(echo $$TASK_ARN | cut -d'/' -f3); \
		echo "Checking log group: /ecs/$(APP)-task"; \
		echo "Checking log stream: ecs/$(APP)-container/$$TASK_ID"; \
		aws logs describe-log-groups \
			--log-group-name-prefix "/ecs/$(APP)-task" \
			--region $(AWS_REGION) \
			--query 'logGroups[0].logGroupName' \
			--output text 2>/dev/null || echo "Log group not found - creating it..."; \
		aws logs create-log-group \
			--log-group-name "/ecs/$(APP)-task" \
			--region $(AWS_REGION) 2>/dev/null || true; \
		aws logs get-log-events \
			--log-group-name "/ecs/$(APP)-task" \
			--log-stream-name "ecs/$(APP)-container/$$TASK_ID" \
			--start-time $$(date -d '2 hours ago' +%s)000 \
			--region $(AWS_REGION) \
			--query 'events[-20:].message' \
			--output text 2>/dev/null || echo "No logs found - container may not have started properly"; \
		echo ""; \
		echo "=== All Log Streams ==="; \
		aws logs describe-log-streams \
			--log-group-name "/ecs/$(APP)-task" \
			--region $(AWS_REGION) \
			--query 'logStreams[].logStreamName' \
			--output text 2>/dev/null || echo "No log streams found"; \
	else \
		echo "No running tasks found"; \
	fi
