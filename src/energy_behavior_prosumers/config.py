import torch
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


class Config:
    """Configuration class for the pipeline."""
    
    def __init__(self, config_file: Optional[str] = None, environment: str = "default"):
        """Initialize configuration from YAML file or defaults."""
        # Load config from YAML if available
        config_data = {}
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                all_configs = yaml.safe_load(f)
                config_data = all_configs.get(environment, all_configs.get('default', {}))
        
        # Set basic configuration
        self.data_dir = config_data.get('data_dir', "data/")
        self.output_dir = config_data.get('output_dir', "output/")
        self.model_dir = config_data.get('model_dir', "models/")
        self.debug = config_data.get('debug', False)
        self.n_day_lags = config_data.get('n_day_lags', 15)
        self.n_estimators = config_data.get('n_estimators', 1500)
        self.early_stopping_rounds = config_data.get('early_stopping_rounds', 100)
        train_block_range_start = config_data.get('train_block_range_start', 0)
        train_block_range_end = config_data.get('train_block_range_end', 600)
        self.train_block_ids = list(range(train_block_range_start, train_block_range_end))
        
        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # MLFlow configuration
        mlflow_config = config_data.get('mlflow', {})
        self.mlflow_tracking_uri = mlflow_config.get('tracking_uri', 'http://localhost:5001')
        self.mlflow_experiment_name = mlflow_config.get('experiment_name', 'energy_behavior_prosumers')
        self.mlflow_artifact_location = mlflow_config.get('artifact_location')
        self.mlflow_registered_model_name = mlflow_config.get('registered_model_name', 'energy_behavior_model')
        self.mlflow_tags = mlflow_config.get('tags', {})
        
        # File paths
        self.files = {
            'train': 'train.csv',
            'client': 'client.csv',
            'historical_weather': 'historical_weather.csv',
            'forecast_weather': 'forecast_weather.csv',
            'electricity': 'electricity_prices.csv',
            'gas': 'gas_prices.csv',
            'location': 'county_lon_lats.csv',
            'county_codes': 'county_id_to_name_map.json'
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'model_dir': self.model_dir,
            'debug': self.debug,
            'n_day_lags': self.n_day_lags,
            'n_estimators': self.n_estimators,
            'early_stopping_rounds': self.early_stopping_rounds,
            'device': self.device,
            'mlflow_tracking_uri': self.mlflow_tracking_uri,
            'mlflow_experiment_name': self.mlflow_experiment_name,
            'mlflow_registered_model_name': self.mlflow_registered_model_name,
        }