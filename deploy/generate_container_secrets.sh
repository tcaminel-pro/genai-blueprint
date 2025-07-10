#!/bin/bash

# Generate secrets array for ECS container definition from .env file
# Usage: ./generate_container_secrets.sh <app-name> <aws-region> <aws-account-id> [env-file-path]

APP=$1
AWS_REGION=$2
AWS_ACCOUNT_ID=$3
DOT_ENV=${4:-.env}

# Debug: Show current directory and check for .env file
echo "Debug: Current directory: $(pwd)" >&2
echo "Debug: Looking for .env file at: $DOT_ENV" >&2
echo "Debug: .env file exists: $(test -f $DOT_ENV && echo 'YES' || echo 'NO')" >&2

if [ ! -f "$DOT_ENV" ]; then
    echo "Warning: .env file not found at $DOT_ENV, generating empty secrets array" >&2
    echo "[]"
    exit 0
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
done < "$DOT_ENV"
echo "]"
echo "Debug: Generated secrets for keys: $(grep -v '^#' "$DOT_ENV" | grep -v '^$' | cut -d'=' -f1 | tr '\n' ' ')" >&2
