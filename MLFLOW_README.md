# MLFlow Integration Guide

This guide covers the MLFlow integration for the Energy Behavior Prosumers prediction project, providing experiment tracking, model registry, and comprehensive model management capabilities.

## Table of Contents

1. [Quick Start](#quick-start)
2. [MLFlow Setup](#mlflow-setup)
3. [Training with MLFlow](#training-with-mlflow)
4. [Model Registry](#model-registry)
5. [Inference with MLFlow](#inference-with-mlflow)
6. [Configuration](#configuration)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Start MLFlow Server

```bash
# Start MLFlow with Docker (MySQL backend)
./scripts/start_mlflow.sh

# Or manually with docker-compose
docker-compose up -d
```

### 2. Install Dependencies

```bash
# Install with uv
uv sync

# Or manually install MLFlow dependencies
pip install mlflow>=2.8.0 pymysql>=1.1.0 pyyaml>=6.0.0
```

### 3. Train a Model with MLFlow Tracking

```bash
# Train with default settings
./scripts/run_pipeline.sh train

# Train in development environment
./scripts/run_pipeline.sh debug

# Train in production environment
./scripts/run_pipeline.sh train --environment production
```

### 4. View Results

- **MLFlow UI**: http://localhost:5001
- **Database Management**: http://localhost:8080 (Adminer)

## MLFlow Setup

### Docker Composition

The project includes a complete MLFlow setup with:

- **MLFlow Server**: Experiment tracking and model registry
- **MySQL Database**: Backend store for experiments and metadata
- **Adminer**: Web-based database management
- **Artifact Storage**: Local filesystem storage (configurable)

### Services

| Service | Port | Purpose |
|---------|------|---------|
| MLFlow Server | 5001 | Main MLFlow UI and API |
| MySQL | 3306 | Backend database |
| Adminer | 8080 | Database management |

### Starting/Stopping

```bash
# Start services
./scripts/start_mlflow.sh

# Stop services
./scripts/stop_mlflow.sh

# View logs
docker-compose logs -f mlflow

# Reset everything (careful: deletes all data)
docker-compose down -v
```

## Training with MLFlow

### Automatic Tracking

When you run training, MLFlow automatically logs:

- **Parameters**: Model hyperparameters, configuration settings
- **Metrics**: Training and validation losses (MAE, MSE, RMSE, R²)
- **Model**: Trained XGBoost model with metadata
- **Artifacts**: Model info, feature lists

### Example Training Commands

```bash
# Basic training
./scripts/run_pipeline.sh train

# Debug training (fast, development environment)
./scripts/run_pipeline.sh debug

# Production training
./scripts/run_pipeline.sh train --environment production --n-estimators 2000

# Custom configuration
./scripts/run_pipeline.sh train --environment production --log-level DEBUG
```

### Logged Information

#### Parameters
- Model hyperparameters (n_estimators, learning_rate, etc.)
- Configuration settings (n_day_lags, data directories)
- Dataset information (train_size, val_size, n_features)

#### Metrics
- Training metrics: `train_mae`, `train_mse`, `train_rmse`, `train_r2`
- Validation metrics: `val_mae`, `val_mse`, `val_rmse`, `val_r2`
- Best iteration and score from XGBoost

#### Model Artifacts
- Trained XGBoost model
- Feature list
- Model metadata (feature count, model type)

## Model Registry

### Automatic Registration

Models are automatically registered in the MLFlow Model Registry after successful training (except in debug mode).

### Model Versions

Each training run creates a new model version with:
- Version number (auto-incrementing)
- Description with key metrics
- Stage (None, Staging, Production, Archived)

### Managing Models via UI

1. Open MLFlow UI: http://localhost:5001
2. Navigate to "Models" tab
3. View registered models and versions
4. Transition models between stages
5. Add descriptions and tags

### Managing Models via Code

```python
from src.energy_behavior_prosumers.config import Config
from src.energy_behavior_prosumers.mlflow_utils import MLFlowManager

config = Config(config_file="config.yaml", environment="production")
mlflow_manager = MLFlowManager(config)

# List all models
models = mlflow_manager.list_models()

# Load specific model
model, features = mlflow_manager.load_model_from_registry(
    model_name="energy_behavior_model",
    version="3"
)
```

## Inference with MLFlow

### Loading Models for Prediction

You have multiple options for loading models during inference:

#### 1. From Model Registry (Recommended)

```bash
# Load latest Production model
./scripts/run_pipeline.sh predict --mlflow-model-stage Production

# Load specific model version
./scripts/run_pipeline.sh predict \
  --mlflow-model-name energy_behavior_model \
  --mlflow-model-version 3

# Load from Staging
./scripts/run_pipeline.sh predict \
  --mlflow-model-name energy_behavior_model \
  --mlflow-model-stage Staging
```

#### 2. From Specific Run

```bash
# Load model from specific MLFlow run
./scripts/run_pipeline.sh predict --mlflow-run-id abc123def456
```

#### 3. From File (Legacy)

```bash
# Load from local file
./scripts/run_pipeline.sh predict --model-path models/xgboost_model.pkl
```

### Example Prediction Workflows

```bash
# Full pipeline with MLFlow model
./scripts/run_pipeline.sh full --mlflow-model-stage Production

# Prediction only with specific version
./scripts/run_pipeline.sh predict \
  --mlflow-model-name energy_behavior_model_prod \
  --mlflow-model-version 5

# Development prediction
./scripts/run_pipeline.sh predict \
  --environment development \
  --mlflow-model-stage Staging
```

## Configuration

### Environment-Specific Configurations

The project supports different environments with MLFlow configurations:

#### Default Environment
```yaml
default:
  mlflow:
    tracking_uri: "http://localhost:5001"
    experiment_name: "energy_behavior_prosumers"
    registered_model_name: "energy_behavior_model"
    tags:
      team: "energy_ml_team"
      project: "energy_behavior_prosumers"
```

#### Development Environment
```yaml
development:
  mlflow:
    tracking_uri: "http://localhost:5001"
    experiment_name: "energy_behavior_prosumers_dev"
    registered_model_name: "energy_behavior_model_dev"
    tags:
      environment: "development"
```

#### Production Environment
```yaml
production:
  mlflow:
    tracking_uri: "http://localhost:5001"  # Change for remote server
    experiment_name: "energy_behavior_prosumers_prod"
    registered_model_name: "energy_behavior_model_prod"
    tags:
      environment: "production"
```

### Remote MLFlow Server

To use a remote MLFlow server, update the `tracking_uri` in your configuration:

```yaml
production:
  mlflow:
    tracking_uri: "https://your-mlflow-server.com"
    # ... other settings
```

## Advanced Usage

### Custom Experiments

```python
from src.energy_behavior_prosumers.config import Config
from src.energy_behavior_prosumers.mlflow_utils import MLFlowManager

config = Config()
mlflow_manager = MLFlowManager(config)

# Start custom run
run_id = mlflow_manager.start_run(
    run_name="custom_experiment",
    tags={"experiment_type": "hyperparameter_tuning"}
)

# Log custom metrics
mlflow_manager.log_metrics({"custom_metric": 0.85})

# End run
mlflow_manager.end_run()
```

### Batch Model Evaluation

```python
# Compare multiple model versions
versions = ["1", "2", "3"]
results = {}

for version in versions:
    model, features = mlflow_manager.load_model_from_registry(
        version=version
    )
    # Evaluate model...
    results[version] = evaluation_metrics
```

### Model Deployment

```python
# Load production model for deployment
model, features = mlflow_manager.load_model_from_registry(
    stage="Production"
)

# Use model for real-time predictions
predictions = model.predict(new_data[features])
```

## Troubleshooting

### Common Issues

#### 1. MLFlow Server Not Starting

```bash
# Check Docker status
docker-compose ps

# View logs
docker-compose logs mlflow

# Restart services
docker-compose restart
```

#### 2. Database Connection Issues

```bash
# Check MySQL logs
docker-compose logs mysql

# Connect to database manually
docker exec -it mlflow_mysql mysql -u mlflow_user -p mlflow_db
```

#### 3. Model Loading Errors

- Ensure MLFlow server is running
- Check model exists in registry
- Verify model stage/version
- Check network connectivity

#### 4. Artifact Storage Issues

```bash
# Check artifacts directory permissions
ls -la mlflow_artifacts/

# Fix permissions if needed
chmod -R 755 mlflow_artifacts/
```

### Configuration Issues

#### Invalid Configuration File

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

#### MLFlow Connection Issues

```bash
# Test MLFlow connection
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5001')
print('MLFlow server status:', mlflow.get_tracking_uri())
"
```

### Performance Optimization

#### 1. Reduce Logging Frequency

Modify the callback in `models.py`:

```python
def callback(env):
    if env.iteration % 50 == 0:  # Log every 50 iterations instead of 10
        # ... logging code
```

#### 2. Disable MLFlow for Large Experiments

```bash
# Train without MLFlow (will still save local model)
export MLFLOW_TRACKING_URI=""
./scripts/run_pipeline.sh train
```

### Migration Guide

#### From Local Files to MLFlow

1. Train new model with MLFlow enabled
2. Register model in registry
3. Test inference with MLFlow model
4. Update production scripts to use MLFlow
5. Archive old model files

#### Upgrading MLFlow

```bash
# Stop services
docker-compose down

# Update MLFlow version in requirements_mlflow.txt
# Restart services
docker-compose up -d
```

## Best Practices

### 1. Experiment Organization

- Use descriptive run names
- Add meaningful tags
- Document experiment purpose
- Use separate experiments for different model types

### 2. Model Lifecycle

- Always test models in Staging before Production
- Use semantic versioning for major changes
- Archive old models when no longer needed
- Document model changes and improvements

### 3. Resource Management

- Clean up unused models and experiments
- Monitor disk usage for artifacts
- Use remote storage for production artifacts
- Regular database maintenance

### 4. Security

- Change default passwords in production
- Use secure connections for remote servers
- Implement access controls
- Regular backups of MLFlow database

## Support

For issues related to MLFlow integration:

1. Check this documentation
2. Review MLFlow official documentation
3. Check project issues on GitHub
4. Contact the team for project-specific questions 