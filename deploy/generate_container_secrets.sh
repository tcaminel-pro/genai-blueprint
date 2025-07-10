#!/bin/bash

# Generate secrets array for ECS container definition from .env file
# Usage: ./generate_container_secrets.sh <app-name> <aws-region> <aws-account-id>

APP=$1
AWS_REGION=$2
AWS_ACCOUNT_ID=$3

if [ ! -f .env ]; then
    echo "Error: .env file not found" >&2
    exit 1
fi

echo -n "["
first=true
while IFS='=' read -r key value; do
    # Skip comments and empty lines
    if [[ $key =~ ^#.*$ ]] || [[ -z "$key" ]]; then
        continue
    fi
    
    if [ "$first" = true ]; then
        first=false
    else
        echo -n ","
    fi
    
    echo -n "{\"name\":\"$key\",\"valueFrom\":\"arn:aws:ssm:$AWS_REGION:$AWS_ACCOUNT_ID:parameter/$APP/$key\"}"
done < .env
echo "]"
