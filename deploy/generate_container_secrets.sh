APP=$1
AWS_REGION=$2
AWS_ACCOUNT_ID=$3
DOT_ENV=${4:-.env}

if [ ! -f "$DOT_ENV" ]; then
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
