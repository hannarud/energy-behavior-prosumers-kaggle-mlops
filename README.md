# Energy Behavior of Prosumers prediction - an MLOps project

Solving ["Enefit - Predict Energy Behavior of Prosumers"](https://www.kaggle.com/c/predict-energy-behavior-of-prosumers) Kaggle competition while extensively using different MLOps techniques. Project created as a part of [MLOps Zoomcamp 2025](https://github.com/DataTalksClub/mlops-zoomcamp) course.

## Table of Contents
1. [Problem Definition](#problem-definition)
2. [Project Outline](#project-outline)
3. [How to Run the Project: Quick Start](#how-to-run-the-project-quick-start)
4. [ML Pipeline Architecture](#ml-pipeline-architecture)
5. [References](#references)

## Problem Definition

Energy prosumers are individuals, businesses, or organizations that both consume and produce energy. This concept represents a shift from the traditional model where consumers simply purchase energy from utilities and rely on centralized power generation sources. Energy prosumers are actively involved in the energy ecosystem by generating their own electricity, typically through renewable energy sources like solar panels (or wind turbines, small-scale hydropower etc.). They also consume energy from the grid when their own generation is insufficient to meet their needs.

<p align="center">
  <img src="https://www.energy.gov/sites/default/files/styles/full_article_width/public/Prosumer-Blog%20sans%20money-%201200%20x%20630-01_0.png?itok=2a3YSkUb" width=600/>
</p>
    
* The number of prosumers is rapidly increasing, associated with higher energy imbalance - increased operational costs, potential grid instability, and inefficient use of energy resources
* The goal of the project is to create an energy prediction model of prosumers to reduce energy imbalance costs. If solved, it would reduce the imbalance costs, improve the reliability of the grid, and make the integration of prosumers into the energy system more efficient and sustainable Moreover, it could potentially incentivize more consumers to become prosumers and thus promote renewable energy production and use

More information on the problem statement, data and other insights for this project can be found [here](https://www.kaggle.com/c/predict-energy-behavior-of-prosumers).

## Project Outline

### ✅ Completed Components

- **[ML Pipeline](src/energy_behavior_prosumers/)** <br> 
Production-ready ML pipeline with feature engineering, model training, and prediction capabilities. Includes comprehensive configuration management and logging.

- **[MLFlow Integration](MLFLOW_README.md)** <br>
Complete MLFlow setup with experiment tracking, model registry, and deployment capabilities. Includes Docker-based server with MySQL backend for production-ready model management.

- **[Notebooks](notebooks/)** <br> 
Exploratory data analysis and model development notebooks for understanding the energy behavior prediction problem.

### 🚧 Planned Components

- **Workflow Orchestration with Prefect** <br>
Orchestration system for automated training, validation, and deployment workflows with scheduling and monitoring capabilities.

- **Model Deployment** <br>
REST API service for real-time energy behavior predictions using models from MLFlow registry, containerized with Docker.

- **Monitoring and Observability** <br>
Model performance monitoring with Evidently, metrics storage in database, and Grafana dashboards for real-time visibility.

- **Testing and Quality Assurance** <br>
Comprehensive test suite with Pytest for pipeline validation, data quality checks, and model performance testing.

## How to Run the Project: Quick Start

Using uv (install it first with `pip install uv`, if needed).

```bash
# 1. Setup with uv
uv sync

# 2. Add your data to data/ directory

# 3. Start MLFlow server (for experiment tracking)
./scripts/start_mlflow.sh

# 4. Run the full ML pipeline with MLFlow tracking
./scripts/run_pipeline.sh full

# 5. For development/debugging
./scripts/run_pipeline.sh debug

# 6. View results in MLFlow UI: http://localhost:5001
```

### MLFlow Integration

This project now includes comprehensive MLFlow integration for:
- **Experiment Tracking**: Automatic logging of parameters, metrics, and models
- **Model Registry**: Centralized model storage and versioning
- **Model Deployment**: Load models from registry for inference

See [MLFLOW_README.md](MLFLOW_README.md) for detailed MLFlow documentation.

## ML Pipeline Architecture

### 📁 **Pipeline Structure**
```
energy-behavior-prosumers-kaggle-mlops/
├── 🧠 Core Pipeline
│   ├── src/energy_behavior_prosumers/
│   │   ├── pipeline.py         # Main ML pipeline
│   │   ├── config.py           # Configuration
│   │   ├── features.py         # Feature engineering
│   │   └── models.py           # Model definitions
│   
├── 🛠️ Utilities
│   ├── scripts/run_pipeline.sh # Bash runner
│   └── config.yaml             # Environment configurations
│   
├── 📈 Outputs
│   ├── models/                 # Trained model checkpoints
│   ├── output/                 # Predictions and submissions
│   └── logs/                   # Pipeline logs
│   
└── 📖 Documentation
    ├── PIPELINE_README.md      # Detailed pipeline guide
    └── USAGE.md                # Usage examples
```

### 🔧 **Pipeline Commands**
```bash
# Full pipeline (training + prediction)
./scripts/run_pipeline.sh full

# Training with MLFlow tracking
./scripts/run_pipeline.sh train --environment production

# Prediction with MLFlow model registry
./scripts/run_pipeline.sh predict --mlflow-model-stage Production

# Prediction with specific MLFlow run
./scripts/run_pipeline.sh predict --mlflow-run-id abc123def456

# Debug mode (fast training)
./scripts/run_pipeline.sh debug
```

### 📊 **MLFlow Commands**
```bash
# Start MLFlow server
./scripts/start_mlflow.sh

# Stop MLFlow server
./scripts/stop_mlflow.sh

# View MLFlow UI: http://localhost:5001
# Database UI: http://localhost:8080
```

### 🔧 Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--mode` | Pipeline execution mode | `full` | `train`, `predict`, `full` |
| `--data-dir` | Input data directory | `data/` | Any valid path |
| `--output-dir` | Output directory | `output/` | Any valid path |
| `--model-dir` | Model checkpoint directory | `models/` | Any valid path |
| `--model-path` | Specific model file path | Auto-generated | Any valid file path |
| `--debug` | Enable debug mode | `False` | Flag |
| `--n-day-lags` | Number of day lags for features | `15` | Any integer ≥ 2 |
| `--n-estimators` | XGBoost estimators | `1500` | Any positive integer |
| `--test-mode` | Test prediction mode | `local` | `local`, `kaggle` |
| `--log-level` | Logging verbosity | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |


## References

- [Mlops Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- [Kaggle competition](https://www.kaggle.com/c/predict-energy-behavior-of-prosumers)

This project heavily benefited from:
* [⚡Enefit│XGBoost│starter⚡ notebook from Kaggle](https://www.kaggle.com/code/rafiko1/enefit-xgboost-starter)
* [Bank Customer Churn Prediction - An MLOps Project from Github](https://github.com/Tobi-Ade/mlops_customer_churn_prediction)
