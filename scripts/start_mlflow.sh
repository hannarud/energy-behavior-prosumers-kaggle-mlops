#!/bin/bash

# Energy Behavior Prosumers - MLFlow Server Startup Script

set -e

echo "🚀 Starting MLFlow Server with Docker..."

# Create artifacts directory if it doesn't exist
mkdir -p mlflow_artifacts

# Start MLFlow services
docker-compose up -d

echo "✅ MLFlow services started!"
echo ""
echo "📊 MLFlow UI: http://localhost:5001"
echo "🗄️  Adminer (Database): http://localhost:8080"
echo "   - Server: mysql"
echo "   - Username: mlflow_user"
echo "   - Password: mlflow_password"
echo "   - Database: mlflow_db"
echo ""
echo "To stop MLFlow services: docker-compose down"
echo "To view logs: docker-compose logs -f mlflow" 