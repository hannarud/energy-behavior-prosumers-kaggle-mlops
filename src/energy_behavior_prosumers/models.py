"""Machine learning models for energy behavior prediction."""

from typing import List, Tuple, Optional
import pandas as pd
import pickle
import xgboost as xgb
import logging
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.energy_behavior_prosumers.config import Config
from src.energy_behavior_prosumers.mlflow_utils import MLFlowManager


class ModelTrainer:
    """XGBoost model training and evaluation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.features = None
        self.mlflow_manager = MLFlowManager(config)
        self.run_id = None
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Prepare training and validation data."""
        # Remove empty target rows
        target = 'target'
        df = df[df[target].notnull()].reset_index(drop=True)
        
        # Split data
        train_data = df[df['data_block_id'].isin(self.config.train_block_ids)]
        val_data = df[~df['data_block_id'].isin(self.config.train_block_ids)]
        
        # Define features
        no_features = ['date', 'latitude', 'longitude', 'data_block_id', 'row_id', 'hours_ahead', 'hour_h']
        remove_columns = [col for col in df.columns for no_feature in no_features if no_feature in col]
        remove_columns.append(target)
        features = [col for col in df.columns if col not in remove_columns]
        
        logging.info(f'Using {len(features)} features for training')
        self.features = features
        
        return train_data, val_data, features
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame, features: List[str]) -> xgb.XGBRegressor:
        """Train XGBoost model with MLFlow tracking."""
        target = 'target'
        
        # Start MLFlow run
        run_name = f"xgboost_training_{self.config.n_estimators}est"
        if self.config.debug:
            run_name += "_debug"
        
        self.run_id = self.mlflow_manager.start_run(
            run_name=run_name,
            tags={"model_type": "xgboost", "debug": str(self.config.debug)}
        )
        
        try:
            # Initialize model
            n_estimators = 2 if self.config.debug else self.config.n_estimators
            
            self.model = xgb.XGBRegressor(
                device=self.config.device,
                enable_categorical=True,
                objective='reg:absoluteerror',
                n_estimators=n_estimators,
                early_stopping_rounds=self.config.early_stopping_rounds,
                random_state=42
            )
            
            # Log parameters
            model_params = self.model.get_params()
            config_params = self.config.to_dict()
            all_params = {**model_params, **config_params}
            all_params.update({
                'n_features': len(features),
                'train_size': len(train_data),
                'val_size': len(val_data)
            })
            self.mlflow_manager.log_params(all_params)
            
            # Train model (without callbacks for compatibility)
            eval_results = {}
            self.model.fit(
                X=train_data[features], 
                y=train_data[target], 
                eval_set=[(train_data[features], train_data[target]), (val_data[features], val_data[target])], 
                verbose=True
            )
            
            # Log intermediate training metrics
            self._log_training_metrics(train_data, val_data, features)
            
            # Calculate final metrics
            train_pred = self.model.predict(train_data[features])
            val_pred = self.model.predict(val_data[features])
            
            # Calculate comprehensive metrics
            train_metrics = self._calculate_metrics(train_data[target], train_pred, "train")
            val_metrics = self._calculate_metrics(val_data[target], val_pred, "val")
            
            # Log final metrics
            final_metrics = {**train_metrics, **val_metrics}
            final_metrics.update({
                'best_iteration': self.model.best_iteration,
                'best_score': self.model.best_score
            })
            self.mlflow_manager.log_metrics(final_metrics)
            
            # Log model
            model_uri = self.mlflow_manager.log_model(self.model, features)
            
            # Register model if not in debug mode
            if not self.config.debug and model_uri:
                version_description = f"XGBoost model with {n_estimators} estimators, MAE: {val_metrics['val_mae']:.4f}"
                self.mlflow_manager.register_model(model_uri, version_description)
            
            logging.info(f'Training completed. Best iteration: {self.model.best_iteration}, '
                        f'Best score: {self.model.best_score:.4f}')
            logging.info(f'Validation MAE: {val_metrics["val_mae"]:.4f}, R²: {val_metrics["val_r2"]:.4f}')
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            self.mlflow_manager.end_run(status="FAILED")
            raise
        finally:
            if self.mlflow_manager.client and not self.config.debug:
                self.mlflow_manager.end_run()
        
        return self.model
    
    def _log_training_metrics(self, train_data: pd.DataFrame, val_data: pd.DataFrame, features: List[str]):
        """Log intermediate training metrics to MLFlow."""
        try:
            if hasattr(self.model, 'evals_result_') and self.model.evals_result_:
                # Get evaluation results from XGBoost
                evals_result = self.model.evals_result_
                
                # Log metrics for each evaluation set
                for eval_name, metrics in evals_result.items():
                    for metric_name, values in metrics.items():
                        # Log the final value and some intermediate values
                        final_value = values[-1]
                        self.mlflow_manager.log_metrics({
                            f"{eval_name}_{metric_name}_final": final_value
                        })
                        
                        # Log every 10th iteration to avoid too much noise
                        for i, value in enumerate(values):
                            if i % 10 == 0 or i == len(values) - 1:
                                self.mlflow_manager.log_metrics({
                                    f"{eval_name}_{metric_name}": value
                                }, step=i)
                                
        except Exception as e:
            logging.warning(f"Could not log training metrics: {e}")
    
    def _calculate_metrics(self, y_true, y_pred, prefix: str) -> dict:
        """Calculate comprehensive metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            f'{prefix}_mae': mae,
            f'{prefix}_mse': mse,
            f'{prefix}_rmse': rmse,
            f'{prefix}_r2': r2
        }
    
    def save_model(self, model_path: str):
        """Save trained model."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'features': self.features,
            'config': self.config.__dict__
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f'Model saved to: {model_path}')
    
    @classmethod
    def load_model(cls, model_path: str, config: Optional[Config] = None):
        """Load trained model from file."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        trainer = cls(config or Config())
        trainer.model = model_data['model']
        trainer.features = model_data['features']
        
        logging.info(f'Model loaded from: {model_path}')
        return trainer
    
    @classmethod
    def load_model_from_mlflow_registry(cls, config: Config, model_name: Optional[str] = None, 
                                       version: Optional[str] = None, stage: str = "Production"):
        """Load model from MLFlow Model Registry."""
        trainer = cls(config)
        model, features = trainer.mlflow_manager.load_model_from_registry(model_name, version, stage)
        
        if model is not None:
            trainer.model = model
            trainer.features = features
            logging.info(f'Model loaded from MLFlow registry')
            return trainer
        else:
            logging.error("Failed to load model from MLFlow registry")
            return None
    
    @classmethod
    def load_model_from_mlflow_run(cls, config: Config, run_id: str, model_name: str = "model"):
        """Load model from specific MLFlow run."""
        trainer = cls(config)
        model, features = trainer.mlflow_manager.load_model_from_run(run_id, model_name)
        
        if model is not None:
            trainer.model = model
            trainer.features = features
            logging.info(f'Model loaded from MLFlow run: {run_id}')
            return trainer
        else:
            logging.error(f"Failed to load model from MLFlow run: {run_id}")
            return None
