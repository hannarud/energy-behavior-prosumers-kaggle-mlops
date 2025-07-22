#!/bin/bash

# Energy Behavior Prosumers - MLFlow Server Startup Script

set -e

echo "🚀 Starting MLFlow Server with Docker (LocalStack S3 + MySQL)..."

# Start MLFlow services
docker-compose up -d

echo "✅ MLFlow services started!"
echo ""
echo "📊 MLFlow UI: http://localhost:5001"
echo "☁️  LocalStack S3: http://localhost:4566"
echo "🗄️  Adminer (Database): http://localhost:8080"
echo "   - Server: mysql"
echo "   - Username: mlflow_user"
echo "   - Password: mlflow_password"
echo "   - Database: mlflow_db"
echo ""
echo "🪣 S3 Artifact Storage:"
echo "   - Bucket: mlflow-artifacts"
echo "   - Endpoint: http://localhost:4566"
echo "   - Region: us-east-1"
echo ""
echo "To stop MLFlow services: docker-compose down"
echo "To view logs: docker-compose logs -f mlflow"
echo "To check LocalStack health: curl http://localhost:4566/_localstack/health" 