"""MLFlow utilities for experiment tracking and model management."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import tempfile
import pickle

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from src.energy_behavior_prosumers.config import Config


class MLFlowManager:
    """MLFlow manager for experiment tracking and model management."""
    
    def __init__(self, config: Config):
        """Initialize MLFlow manager with configuration."""
        self.config = config
        self.client = None
        self.experiment_id = None
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLFlow tracking and experiment."""
        try:
            # Configure S3 endpoint for LocalStack if provided
            if self.config.mlflow_s3_endpoint_url:
                import os
                os.environ['MLFLOW_S3_ENDPOINT_URL'] = self.config.mlflow_s3_endpoint_url
                logging.info(f"S3 endpoint URL configured: {self.config.mlflow_s3_endpoint_url}")
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            self.client = MlflowClient(self.config.mlflow_tracking_uri)
            
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.config.mlflow_experiment_name)
                if experiment is None:
                    self.experiment_id = mlflow.create_experiment(
                        name=self.config.mlflow_experiment_name,
                        artifact_location=self.config.mlflow_artifact_location
                    )
                else:
                    self.experiment_id = experiment.experiment_id
            except Exception as e:
                logging.warning(f"Could not create/get experiment: {e}")
                self.experiment_id = mlflow.create_experiment(
                    name=f"{self.config.mlflow_experiment_name}_fallback"
                )
            
            # Set experiment
            mlflow.set_experiment(experiment_id=self.experiment_id)
            
            logging.info(f"MLFlow setup complete. Tracking URI: {self.config.mlflow_tracking_uri}")
            logging.info(f"Experiment: {self.config.mlflow_experiment_name} (ID: {self.experiment_id})")
            
        except Exception as e:
            logging.error(f"Failed to setup MLFlow: {e}")
            logging.warning("MLFlow tracking will be disabled")
            self.client = None
            self.experiment_id = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Start an MLFlow run."""
        if not self.client:
            logging.warning("MLFlow not available, skipping run start")
            return None
        
        try:
            # Combine default tags with provided tags
            all_tags = self.config.mlflow_tags.copy()
            if tags:
                all_tags.update(tags)
            
            run = mlflow.start_run(run_name=run_name, tags=all_tags)
            logging.info(f"Started MLFlow run: {run.info.run_id}")
            return run.info.run_id
        except Exception as e:
            logging.error(f"Failed to start MLFlow run: {e}")
            return None
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLFlow."""
        if not mlflow.active_run():
            logging.warning("No active MLFlow run, skipping parameter logging")
            return
        
        try:
            # Filter out non-serializable parameters
            serializable_params = {}
            for key, value in params.items():
                try:
                    # Convert to string if not a basic type
                    if isinstance(value, (str, int, float, bool)):
                        serializable_params[key] = value
                    else:
                        serializable_params[key] = str(value)
                except Exception:
                    serializable_params[key] = str(type(value).__name__)
            
            mlflow.log_params(serializable_params)
            logging.info(f"Logged {len(serializable_params)} parameters")
        except Exception as e:
            logging.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLFlow."""
        if not mlflow.active_run():
            logging.warning("No active MLFlow run, skipping metrics logging")
            return
        
        try:
            mlflow.log_metrics(metrics, step=step)
            logging.info(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            logging.error(f"Failed to log metrics: {e}")
    
    def log_model(self, model, features: list, model_name: str = "model") -> Optional[str]:
        """Log model to MLFlow."""
        if not mlflow.active_run():
            logging.warning("No active MLFlow run, skipping model logging")
            return None
        
        try:
            # Create model info dictionary
            model_info = {
                'features': features,
                'n_features': len(features),
                'model_type': type(model).__name__
            }
            
            # Log XGBoost model
            if hasattr(model, 'predict') and hasattr(model, 'get_params'):
                model_uri = mlflow.xgboost.log_model(
                    model,
                    model_name,
                    signature=None,  # Will be inferred
                    input_example=None,
                    pip_requirements=["xgboost", "pandas", "numpy"]
                ).model_uri
                
                # Log additional model info as artifacts
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    import json
                    json.dump(model_info, f, indent=2)
                    mlflow.log_artifact(f.name, "model_info.json")
                    Path(f.name).unlink()  # Clean up temp file
                
                logging.info(f"Model logged successfully: {model_uri}")
                return model_uri
            else:
                logging.warning("Model type not supported for MLFlow logging")
                return None
                
        except Exception as e:
            logging.error(f"Failed to log model: {e}")
            return None
    
    def register_model(self, model_uri: str, version_description: Optional[str] = None) -> Optional[str]:
        """Register model in MLFlow Model Registry."""
        if not self.client or not model_uri:
            logging.warning("MLFlow not available or no model URI provided")
            return None
        
        try:
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=self.config.mlflow_registered_model_name
            )
            
            # Add description if provided
            if version_description:
                self.client.update_model_version(
                    name=self.config.mlflow_registered_model_name,
                    version=model_version.version,
                    description=version_description
                )
            
            logging.info(f"Model registered: {self.config.mlflow_registered_model_name} v{model_version.version}")
            return f"{self.config.mlflow_registered_model_name}/{model_version.version}"
            
        except Exception as e:
            logging.error(f"Failed to register model: {e}")
            return None
    
    def load_model_from_registry(self, model_name: Optional[str] = None, version: Optional[str] = None, 
                                stage: str = "Production") -> Tuple[Optional[Any], Optional[list]]:
        """Load model from MLFlow Model Registry."""
        if not self.client:
            logging.warning("MLFlow not available")
            return None, None
        
        try:
            model_name = model_name or self.config.mlflow_registered_model_name
            
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/{stage}"
            
            # Load model
            model = mlflow.xgboost.load_model(model_uri)
            
            # Try to load features info
            features = None
            try:
                # Download artifacts to get features info
                artifact_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"{model_uri}/model_info.json"
                )
                if Path(artifact_path).exists():
                    import json
                    with open(artifact_path) as f:
                        model_info = json.load(f)
                        features = model_info.get('features')
            except Exception:
                logging.warning("Could not load features information from model artifacts")
            
            logging.info(f"Model loaded from registry: {model_uri}")
            return model, features
            
        except Exception as e:
            logging.error(f"Failed to load model from registry: {e}")
            return None, None
    
    def load_model_from_run(self, run_id: str, model_name: str = "model") -> Tuple[Optional[Any], Optional[list]]:
        """Load model from specific MLFlow run."""
        if not self.client:
            logging.warning("MLFlow not available")
            return None, None
        
        try:
            model_uri = f"runs:/{run_id}/{model_name}"
            
            # Load model
            model = mlflow.xgboost.load_model(model_uri)
            
            # Try to load features info
            features = None
            try:
                # Download artifacts to get features info
                artifact_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"{model_uri}/model_info.json"
                )
                if Path(artifact_path).exists():
                    import json
                    with open(artifact_path) as f:
                        model_info = json.load(f)
                        features = model_info.get('features')
            except Exception:
                logging.warning("Could not load features information from model artifacts")
            
            logging.info(f"Model loaded from run: {model_uri}")
            return model, features
            
        except Exception as e:
            logging.error(f"Failed to load model from run: {e}")
            return None, None
    
    def end_run(self, status: str = "FINISHED"):
        """End current MLFlow run."""
        if mlflow.active_run():
            try:
                mlflow.end_run(status=status)
                logging.info("MLFlow run ended")
            except Exception as e:
                logging.error(f"Failed to end MLFlow run: {e}")
    
    def list_models(self) -> list:
        """List all registered models."""
        if not self.client:
            return []
        
        try:
            models = self.client.list_registered_models()
            return [(model.name, model.latest_versions) for model in models]
        except Exception as e:
            logging.error(f"Failed to list models: {e}")
            return [] 