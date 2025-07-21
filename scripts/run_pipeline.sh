#!/bin/bash

# Energy Behavior Prosumers ML Pipeline
# Run the complete machine learning pipeline with MLFlow support

set -e

# Default values
MODE="full"
DEBUG=false
LOG_LEVEL="INFO"
ENVIRONMENT="default"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        train|predict|full)
            MODE="$1"
            shift
            ;;
        debug)
            DEBUG=true
            ENVIRONMENT="development"
            shift
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --n-estimators)
            N_ESTIMATORS="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --mlflow-run-id)
            MLFLOW_RUN_ID="$2"
            shift 2
            ;;
        --mlflow-model-name)
            MLFLOW_MODEL_NAME="$2"
            shift 2
            ;;
        --mlflow-model-version)
            MLFLOW_MODEL_VERSION="$2"
            shift 2
            ;;
        --mlflow-model-stage)
            MLFLOW_MODEL_STAGE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [train|predict|full|debug] [options]"
            echo ""
            echo "Commands:"
            echo "  train    Run training only"
            echo "  predict  Run prediction only" 
            echo "  full     Run full pipeline (default)"
            echo "  debug    Run in debug mode (uses development environment)"
            echo ""
            echo "Options:"
            echo "  --environment ENV            Configuration environment (default, development, production)"
            echo "  --n-estimators NUM           Number of XGBoost estimators"
            echo "  --model-path PATH            Path to model file (legacy loading)"
            echo "  --mlflow-run-id ID           MLFlow run ID to load model from"
            echo "  --mlflow-model-name NAME     MLFlow registered model name"
            echo "  --mlflow-model-version VER   MLFlow model version"
            echo "  --mlflow-model-stage STAGE   MLFlow model stage (Production, Staging, etc.)"
            echo "  --log-level LEVEL            Logging level (DEBUG, INFO, WARNING, ERROR)"
            echo "  -h, --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 train --environment production"
            echo "  $0 predict --mlflow-model-name energy_behavior_model --mlflow-model-stage Production"
            echo "  $0 predict --mlflow-run-id abc123def456"
            echo "  $0 debug"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="uv run python -m src.energy_behavior_prosumers.pipeline"
CMD="$CMD --mode $MODE"
CMD="$CMD --environment $ENVIRONMENT"
CMD="$CMD --log-level $LOG_LEVEL"

if [ "$DEBUG" = true ]; then
    CMD="$CMD --debug"
fi

if [ -n "$N_ESTIMATORS" ]; then
    CMD="$CMD --n-estimators $N_ESTIMATORS"
fi

if [ -n "$MODEL_PATH" ]; then
    CMD="$CMD --model-path $MODEL_PATH"
fi

if [ -n "$MLFLOW_RUN_ID" ]; then
    CMD="$CMD --mlflow-run-id $MLFLOW_RUN_ID"
fi

if [ -n "$MLFLOW_MODEL_NAME" ]; then
    CMD="$CMD --mlflow-model-name $MLFLOW_MODEL_NAME"
fi

if [ -n "$MLFLOW_MODEL_VERSION" ]; then
    CMD="$CMD --mlflow-model-version $MLFLOW_MODEL_VERSION"
fi

if [ -n "$MLFLOW_MODEL_STAGE" ]; then
    CMD="$CMD --mlflow-model-stage $MLFLOW_MODEL_STAGE"
fi

echo "🚀 Running Energy Behavior Prosumers Pipeline..."
echo "Mode: $MODE"
echo "Environment: $ENVIRONMENT"
if [ "$DEBUG" = true ]; then
    echo "Debug: Enabled"
fi
echo "Command: $CMD"
echo ""

# Run the pipeline
eval $CMD

echo ""
echo "✅ Pipeline completed successfully!" 