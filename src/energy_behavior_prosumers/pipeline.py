#!/usr/bin/env python3
"""
Energy Behavior Prosumers ML Pipeline
Converts the notebook workflow into a production-ready pipeline.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from colorama import Fore, Style

from src.energy_behavior_prosumers.features import FeatureProcessor, TargetProcessor
from src.energy_behavior_prosumers.models import ModelTrainer
from src.energy_behavior_prosumers.config import Config


class Logger:
    """Logging utility."""
    
    @staticmethod
    def setup_logging(log_level: str = 'INFO'):
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    @staticmethod
    def print_color(text: str, color=Fore.BLUE, style=Style.BRIGHT):
        """Print colored text."""
        print(style + color + text + Style.RESET_ALL)


class DataLoader:
    """Data loading utilities."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all required datasets."""
        data = {}
        
        # Load CSV files
        csv_files = ['train', 'client', 'historical_weather', 'forecast_weather', 'electricity', 'gas', 'location']
        for file_key in csv_files:
            file_path = Path(self.config.data_dir) / self.config.files[file_key]
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
            data[file_key] = pd.read_csv(file_path)
            logging.info(f"Loaded {file_key}: {data[file_key].shape}")
        
        # Load JSON file
        county_codes_path = Path(self.config.data_dir) / self.config.files['county_codes']
        if county_codes_path.exists():
            with open(county_codes_path) as f:
                data['county_codes'] = json.load(f)
        
        # Clean location data
        if 'Unnamed: 0' in data['location'].columns:
            data['location'] = data['location'].drop(columns=["Unnamed: 0"])
        
        return data


class PredictionPipeline:
    """Handle prediction pipeline for test data."""
    
    def __init__(self, config: Config, model_trainer: ModelTrainer):
        self.config = config
        self.model_trainer = model_trainer
        self.feature_processor = FeatureProcessor()
        self.target_processor = TargetProcessor()
        self.location_data = self._load_location_data()
    
    def _load_location_data(self) -> pd.DataFrame:
        """Load location data if available."""
        try:
            location_path = Path(self.config.data_dir) / self.config.files['location']
            if location_path.exists():
                location = pd.read_csv(location_path)
                # Clean location data
                if 'Unnamed: 0' in location.columns:
                    location = location.drop(columns=["Unnamed: 0"])
                logging.info(f"Loaded location data: {location.shape}")
                return location
            else:
                logging.warning(f"Location file not found: {location_path}")
                return pd.DataFrame()
        except Exception as e:
            logging.warning(f"Could not load location data: {e}")
            return pd.DataFrame()
    
    def run_prediction(self, test_mode: str = 'local') -> Optional[pd.DataFrame]:
        """Run prediction pipeline."""
        if test_mode == 'kaggle':
            return self._run_kaggle_prediction()
        else:
            return self._run_local_prediction()
    
    def _run_local_prediction(self) -> pd.DataFrame:
        """Run prediction on local test data (if available)."""
        logging.info("Local prediction mode - this would run on test data if available")
        # This would be implemented based on available test data format
        return pd.DataFrame()
    
    def _run_kaggle_prediction(self) -> Optional[pd.DataFrame]:
        """Run prediction using Kaggle competition environment."""
        try:
            import enefit
            env = enefit.make_env()
            iter_test = env.iter_test()
            
            # Reload enefit environment (only in debug mode)
            if self.config.debug:
                enefit.make_env.__called__ = False
                type(env)._state = type(type(env)._state).__dict__['INIT']
                iter_test = env.iter_test()
            
            previous_revealed_targets = []
            
            for (test, revealed_targets, client_test, historical_weather_test,
                 forecast_weather_test, electricity_test, gas_test, sample_prediction) in iter_test:
                
                # Rename test set to make consistent with train
                test = test.rename(columns={'prediction_datetime': 'datetime'})

                # Initialize data_block_id column
                id_column = 'data_block_id' 
                for data in [test, gas_test, electricity_test, historical_weather_test, 
                           forecast_weather_test, client_test, revealed_targets]:
                    data[id_column] = 0
                
                # Process features
                data_test = self.feature_processor.process_features(
                    data=test,
                    client=client_test, 
                    historical_weather=historical_weather_test,
                    forecast_weather=forecast_weather_test, 
                    electricity=electricity_test, 
                    gas=gas_test,
                    location=self.location_data
                )
                
                # Store revealed_targets
                previous_revealed_targets.insert(0, revealed_targets)
                
                if len(previous_revealed_targets) == self.config.n_day_lags:
                    previous_revealed_targets.pop()
                
                # Add previous revealed targets
                df_test = self.target_processor.create_revealed_targets_test(
                    data=data_test.copy(),
                    previous_revealed_targets=previous_revealed_targets.copy(),
                    n_day_lags=self.config.n_day_lags
                )
                
                # Make prediction
                X_test = df_test[self.model_trainer.features]
                sample_prediction['target'] = self.model_trainer.model.predict(X_test)
                env.predict(sample_prediction)
            
            logging.info("Kaggle prediction completed successfully")
            return None
            
        except ImportError:
            logging.warning("enefit module not available - skipping Kaggle prediction")
            return None


class MLPipeline:
    """Main ML Pipeline orchestrator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.feature_processor = FeatureProcessor()
        self.target_processor = TargetProcessor()
        self.model_trainer = ModelTrainer(config)
    
    def run_training_pipeline(self) -> ModelTrainer:
        """Run the complete training pipeline."""
        logging.info("Starting ML training pipeline...")
        
        # Load data
        data = self.data_loader.load_data()
        
        # Process features
        df = self.feature_processor.process_features(
            data=data['train'].copy(),
            client=data['client'].copy(),
            historical_weather=data['historical_weather'].copy(),
            forecast_weather=data['forecast_weather'].copy(),
            electricity=data['electricity'].copy(),
            gas=data['gas'].copy(),
            location=data['location'].copy()
        )
        
        # Add revealed targets
        df = self.target_processor.create_revealed_targets_train(
            df.copy(), 
            n_day_lags=self.config.n_day_lags
        )
        
        # Prepare data and train model
        train_data, val_data, features = self.model_trainer.prepare_data(df)
        self.model_trainer.train(train_data, val_data, features)
        
        # Save model
        model_path = Path(self.config.model_dir) / "xgboost_model.pkl"
        self.model_trainer.save_model(str(model_path))
        
        logging.info("Training pipeline completed successfully")
        return self.model_trainer
    
    def run_prediction_pipeline(self, model_path: Optional[str] = None, test_mode: str = 'local') -> Optional[pd.DataFrame]:
        """Run the prediction pipeline."""
        logging.info("Starting prediction pipeline...")
        
        # Load model if path provided
        if model_path:
            self.model_trainer = ModelTrainer.load_model(model_path)
        
        # Run predictions
        prediction_pipeline = PredictionPipeline(self.config, self.model_trainer)
        results = prediction_pipeline.run_prediction(test_mode)
        
        logging.info("Prediction pipeline completed")
        return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Energy Behavior Prosumers ML Pipeline')
    
    parser.add_argument('--mode', choices=['train', 'predict', 'full'], default='full',
                       help='Pipeline mode: train only, predict only, or full pipeline')
    parser.add_argument('--data-dir', default='data/', 
                       help='Directory containing input data files')
    parser.add_argument('--output-dir', default='output/',
                       help='Directory for output files')
    parser.add_argument('--model-dir', default='models/',
                       help='Directory for model checkpoints')
    parser.add_argument('--model-path', type=str,
                       help='Path to saved model (for prediction mode)')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with reduced iterations')
    parser.add_argument('--n-day-lags', type=int, default=15,
                       help='Number of day lags for revealed targets')
    parser.add_argument('--n-estimators', type=int, default=1500,
                       help='Number of XGBoost estimators')
    parser.add_argument('--test-mode', choices=['local', 'kaggle'], default='local',
                       help='Test prediction mode')
    parser.add_argument('--log-level', default='INFO',
                       help='Logging level')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    Logger.setup_logging(args.log_level)
    
    # Create config
    config = Config()
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.model_dir = args.model_dir
    config.debug = args.debug
    config.n_day_lags = args.n_day_lags
    config.n_estimators = args.n_estimators
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = MLPipeline(config)
    
    try:
        if args.mode in ['train', 'full']:
            # Run training
            model_trainer = pipeline.run_training_pipeline()
            
        if args.mode in ['predict', 'full']:
            # Run prediction
            model_path = args.model_path if args.model_path else str(Path(config.model_dir) / "xgboost_model.pkl")
            results = pipeline.run_prediction_pipeline(model_path, args.test_mode)
            
            if results is not None and not results.empty:
                output_path = Path(config.output_dir) / "submission.csv"
                results.to_csv(output_path, index=False)
                logging.info(f"Predictions saved to: {output_path}")
        
        logging.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
