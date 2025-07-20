"""Machine learning models for energy behavior prediction."""

from typing import List, Tuple
import pandas as pd
import pickle
import xgboost as xgb
import logging
import os

from src.energy_behavior_prosumers.config import Config


class ModelTrainer:
    """XGBoost model training and evaluation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.features = None
    
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
        """Train XGBoost model."""
        target = 'target'
        
        # Initialize model
        n_estimators = 2 if self.config.debug else self.config.n_estimators
        
        self.model = xgb.XGBRegressor(
            device=self.config.device,
            enable_categorical=True,
            objective='reg:absoluteerror',
            n_estimators=n_estimators,
            early_stopping_rounds=self.config.early_stopping_rounds
        )
        
        # Train model
        self.model.fit(
            X=train_data[features], 
            y=train_data[target], 
            eval_set=[(train_data[features], train_data[target]), (val_data[features], val_data[target])], 
            verbose=False
        )
        
        logging.info(f'Training completed. Best iteration: {self.model.best_iteration}, '
                    f'Best score: {self.model.best_score:.4f}')
        
        return self.model
    
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
    def load_model(cls, model_path: str):
        """Load trained model."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        trainer = cls(Config())
        trainer.model = model_data['model']
        trainer.features = model_data['features']
        
        logging.info(f'Model loaded from: {model_path}')
        return trainer
