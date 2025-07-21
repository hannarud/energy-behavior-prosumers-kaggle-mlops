#!/bin/bash

# Energy Behavior Prosumers - MLFlow Server Stop Script

set -e

echo "🛑 Stopping MLFlow Server..."

# Stop MLFlow services
docker-compose down

echo "✅ MLFlow services stopped!" 