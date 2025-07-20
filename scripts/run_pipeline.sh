#!/bin/bash

# Energy Behavior Prosumers ML Pipeline Runner
# Convenience script for running common pipeline operations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if required directories exist
check_directories() {
    print_status "Checking required directories..."
    
    if [ ! -d "data" ]; then
        print_warning "data/ directory not found. Creating it..."
        mkdir -p data
    fi
    
    if [ ! -d "output" ]; then
        mkdir -p output
        print_status "Created output/ directory"
    fi
    
    if [ ! -d "models" ]; then
        mkdir -p models
        print_status "Created models/ directory"
    fi
}

# Function to check if Python dependencies are installed
check_dependencies() {
    print_status "Checking Python dependencies..."
    
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check if uv is available
    if command -v uv &> /dev/null; then
        print_status "uv is available - checking environment sync..."
        if ! uv sync --check 2>/dev/null; then
            print_warning "Dependencies not synced. Run: uv sync"
        fi
    else
        print_warning "uv not found, checking packages manually..."
    fi
    
    # Check if required packages are installed
    python -c "import pandas, numpy, xgboost, torch" 2>/dev/null || {
        print_error "Required Python packages not found. Please run: uv sync"
        exit 1
    }
    
    print_success "All dependencies are available"
}

# Function to validate data files
validate_data() {
    print_status "Validating data files..."
    
    required_files=(
        "data/train.csv"
        "data/client.csv"
        "data/historical_weather.csv"
        "data/forecast_weather.csv"
        "data/electricity_prices.csv"
        "data/gas_prices.csv"
        "data/county_lon_lats.csv"
        "data/county_id_to_name_map.json"
    )
    
    missing_files=()
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        print_error "Missing required data files:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        print_warning "Please ensure all required data files are in the data/ directory"
        exit 1
    fi
    
    print_success "All required data files found"
}

# Function to run full pipeline
run_full_pipeline() {
    print_status "Running full ML pipeline (training + prediction)..."
    python -m src.energy_behavior_prosumers.pipeline \
        --mode full \
        --data-dir data/ \
        --output-dir output/ \
        --model-dir models/ \
        --log-level INFO
    print_success "Full pipeline completed successfully!"
}

# Function to run training only
run_training() {
    print_status "Running training pipeline..."
    python -m src.energy_behavior_prosumers.pipeline \
        --mode train \
        --data-dir data/ \
        --model-dir models/ \
        --n-estimators ${N_ESTIMATORS:-1500} \
        --log-level INFO
    print_success "Training completed successfully!"
}

# Function to run prediction only
run_prediction() {
    local model_path=${1:-"models/xgboost_model.pkl"}
    
    if [ ! -f "$model_path" ]; then
        print_error "Model file not found: $model_path"
        print_warning "Please run training first or specify a valid model path"
        exit 1
    fi
    
    print_status "Running prediction pipeline with model: $model_path"
    python -m src.energy_behavior_prosumers.pipeline \
        --mode predict \
        --model-path "$model_path" \
        --output-dir output/ \
        --test-mode local \
        --log-level INFO
    print_success "Prediction completed successfully!"
}

# Function to run debug/development mode
run_debug() {
    print_status "Running pipeline in debug mode (fast training)..."
    python -m src.energy_behavior_prosumers.pipeline \
        --mode full \
        --debug \
        --data-dir data/ \
        --output-dir output/debug/ \
        --model-dir models/debug/ \
        --n-estimators 10 \
        --log-level DEBUG
    print_success "Debug run completed successfully!"
}

# Function to run Kaggle competition mode
run_kaggle() {
    print_status "Running pipeline in Kaggle competition mode..."
    python -m src.energy_behavior_prosumers.pipeline \
        --mode predict \
        --test-mode kaggle \
        --model-path models/xgboost_model.pkl \
        --log-level INFO
    print_success "Kaggle prediction completed successfully!"
}

# Function to clean outputs
clean_outputs() {
    print_status "Cleaning output directories..."
    rm -rf output/*
    rm -rf models/*
    print_success "Output directories cleaned"
}

# Function to show help
show_help() {
    echo "Energy Behavior Prosumers ML Pipeline Runner"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  full        Run complete pipeline (training + prediction)"
    echo "  train       Run training only"
    echo "  predict     Run prediction only (requires trained model)"
    echo "  debug       Run in debug mode (fast training for testing)"
    echo "  kaggle      Run in Kaggle competition mode"
    echo "  validate    Validate data files and dependencies"
    echo "  clean       Clean output and model directories"
    echo "  help        Show this help message"
    echo ""
    echo "Options:"
    echo "  --model-path PATH    Specify model path for prediction (default: models/xgboost_model.pkl)"
    echo "  --n-estimators N     Number of XGBoost estimators for training (default: 1500)"
    echo ""
    echo "Examples:"
    echo "  $0 full                           # Run complete pipeline"
    echo "  $0 train --n-estimators 2000     # Train with 2000 estimators"
    echo "  $0 predict --model-path my_model.pkl  # Predict with specific model"
    echo "  $0 debug                          # Quick debug run"
    echo ""
}

# Parse command line arguments
COMMAND=$1
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --n-estimators)
            N_ESTIMATORS="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Execute based on command
case $COMMAND in
    full)
        check_directories
        check_dependencies
        validate_data
        run_full_pipeline
        ;;
    train)
        check_directories
        check_dependencies
        validate_data
        run_training
        ;;
    predict)
        check_directories
        check_dependencies
        run_prediction "$MODEL_PATH"
        ;;
    debug)
        check_directories
        check_dependencies
        validate_data
        run_debug
        ;;
    kaggle)
        check_dependencies
        run_kaggle
        ;;
    validate)
        check_directories
        check_dependencies
        validate_data
        print_success "All validations passed!"
        ;;
    clean)
        clean_outputs
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        print_warning "No command specified. Running full pipeline..."
        check_directories
        check_dependencies
        validate_data
        run_full_pipeline
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac 