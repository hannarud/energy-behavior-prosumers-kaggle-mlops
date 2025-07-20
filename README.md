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
- [notebooks](https://github.com/hannarud/energy-behavior-prosumers-kaggle-mlops/tree/main/notebooks) <br> 
Here we create notebooks where we clean and preprocess the data, before training our ml models. We train different models and select the best performing model. <br>
We spend minimal time building the model so as to focus on the major goal of the project (MLOps). 

- [MLFlow and Model Registry](https://github.com/Tobi-Ade/mlops_customer_churn_prediction/blob/main/notebooks/churn_prediction_mlflow.ipynb) <br>
We use MLFlow and Model Registry to track the performance of our models. This gives us a structured way to monitor and store our artifacts (models, preprocessors, etc).
We access our stored artifacts multiple times to train new models.

- [Workflow Orchestration with Prefect](https://github.com/Tobi-Ade/mlops_customer_churn_prediction/tree/main/workflow-orchestration) <br>
Prefect allows us to set a defined structure for our project. We can deploy our project and use work queues to track our deployments and run them whenever we wanr as prefect  flows. <br>
The prefect ui also provides an interface to see our flow runs and logs from every run.

- [Deployment](https://github.com/Tobi-Ade/mlops_customer_churn_prediction/tree/main/deployment) <br>
Here we deploy our model as a web service using flask. We also use docker o add an eextra layer of isolation for the web service. We build a version of the service using our artifacts stored in model registry.

- [Monitoring with Evidently, Grafana, Adminer]() <br>
We use evidently to track how model performance by defining metrics on our data. Adminer to manage por postgres database where store our metrics from evidently. <br>
Then we use Grafana to pull these data metrics and create dashboards to make monitoring these metrics easier.

- [Best-Practices]() <br>
Here we create tests for our scripts to make sure the result of every run is exactly what we intended. We use Pytest, DeepDiff, and Make to run and automate the tests.

## How to Run the Project: Quick Start

Using uv (install it first with `pip install uv`, if needed).

```bash
# 1. Setup with uv
uv sync

# 2. Add your data to data/ directory

# 3. Run the full ML pipeline
uv run python -m src.energy_behavior_prosumers.pipeline
# OR
uv run ./scripts/run_pipeline.sh full

# 4. For development/debugging
uv run ./scripts/run_pipeline.sh debug
```

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

# Training only with custom parameters
./scripts/run_pipeline.sh train --n-estimators 2000

# Prediction with specific model
./scripts/run_pipeline.sh predict --model-path models/my_model.pkl

# Debug mode (fast training)
./scripts/run_pipeline.sh debug
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
