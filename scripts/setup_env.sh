#!/bin/bash

# Environment setup for LocalStack S3 integration with MLflow
# Source this file before running your Python scripts

export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_S3_ENDPOINT_URL=http://localhost:4566

echo "✅ Environment variables set for LocalStack S3:"
echo "   AWS_ACCESS_KEY_ID=test"
echo "   AWS_SECRET_ACCESS_KEY=test"
echo "   AWS_DEFAULT_REGION=us-east-1"
echo "   MLFLOW_S3_ENDPOINT_URL=http://localhost:4566"
echo ""
echo "💡 Usage:"
echo "   Manual setup: source scripts/setup_env.sh"
echo "   Automatic:    ./scripts/run_pipeline.sh (includes this setup)"
echo ""
echo "🏃 Quick start:"
echo "   ./scripts/start_mlflow.sh    # Start MLflow + LocalStack"
echo "   ./scripts/run_pipeline.sh    # Run pipeline with auto-setup" 