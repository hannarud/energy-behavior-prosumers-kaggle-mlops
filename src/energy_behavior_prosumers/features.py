"""Feature engineering for energy behavior prediction."""

import pandas as pd
import numpy as np
from typing import List


class FeatureProcessor:
    """Feature processing class."""
    
    def __init__(self):         
        # Columns to join on for the different datasets
        self.weather_join = ['datetime', 'county', 'data_block_id']
        self.gas_join = ['data_block_id']
        self.electricity_join = ['datetime', 'data_block_id']
        self.client_join = ['county', 'is_business', 'product_type', 'data_block_id']
        
        # Columns of latitude & longitude
        self.lat_lon_columns = ['latitude', 'longitude']
        
        # Aggregate stats 
        self.agg_stats = ['mean']
        
        # Categorical columns (specify for XGBoost)
        self.category_columns = ['county', 'is_business', 'product_type', 'is_consumption', 'data_block_id']

    def create_new_column_names(self, df: pd.DataFrame, suffix: str, columns_no_change: List[str]) -> pd.DataFrame:
        """Change column names by given suffix, keep columns_no_change."""
        df.columns = [col + suffix if col not in columns_no_change else col for col in df.columns]
        return df 

    def flatten_multi_index_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten multi-index columns."""
        df.columns = ['_'.join([col for col in multi_col if len(col) > 0]) for multi_col in df.columns]
        return df
    
    def create_data_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for main data (test or train) set."""
        # To datetime
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        # Time period features
        data['date'] = data['datetime'].dt.normalize()
        data['year'] = data['datetime'].dt.year
        data['quarter'] = data['datetime'].dt.quarter
        data['month'] = data['datetime'].dt.month
        data['week'] = data['datetime'].dt.isocalendar().week
        data['hour'] = data['datetime'].dt.hour
        
        # Day features
        data['day_of_year'] = data['datetime'].dt.day_of_year
        data['day_of_month'] = data['datetime'].dt.day
        data['day_of_week'] = data['datetime'].dt.day_of_week
        return data

    def create_client_features(self, client: pd.DataFrame) -> pd.DataFrame:
        """Create client features."""
        client = self.create_new_column_names(
            client, 
            suffix='_client',
            columns_no_change=self.client_join
        )       
        return client
    
    def create_historical_weather_features(self, historical_weather: pd.DataFrame, location: pd.DataFrame) -> pd.DataFrame:
        """Create historical weather features."""
        # To datetime
        historical_weather['datetime'] = pd.to_datetime(historical_weather['datetime'])
        
        # Add county
        historical_weather[self.lat_lon_columns] = historical_weather[self.lat_lon_columns].astype(float).round(1)
        historical_weather = historical_weather.merge(location, how='left', on=self.lat_lon_columns)

        # Modify column names - specify suffix
        historical_weather = self.create_new_column_names(
            historical_weather,
            suffix='_h',
            columns_no_change=self.lat_lon_columns + self.weather_join
        ) 
        
        # Group by & calculate aggregate stats 
        agg_columns = [col for col in historical_weather.columns if col not in self.lat_lon_columns + self.weather_join]
        agg_dict = {agg_col: self.agg_stats for agg_col in agg_columns}
        historical_weather = historical_weather.groupby(self.weather_join).agg(agg_dict).reset_index() 
        
        # Flatten the multi column aggregates
        historical_weather = self.flatten_multi_index_columns(historical_weather) 
        
        # Test set has 1 day offset for hour<11 and 2 day offset for hour>11
        historical_weather['hour_h'] = historical_weather['datetime'].dt.hour
        historical_weather['datetime'] = (
            historical_weather.apply(
                lambda x: x['datetime'] + pd.DateOffset(1) 
                if x['hour_h'] < 11 
                else x['datetime'] + pd.DateOffset(2),
                axis=1
            )
        )
        
        return historical_weather
    
    def create_forecast_weather_features(self, forecast_weather: pd.DataFrame, location: pd.DataFrame) -> pd.DataFrame:
        """Create forecast weather features."""
        # Rename column and drop
        forecast_weather = (
            forecast_weather
            .rename(columns={'forecast_datetime': 'datetime'})
            .drop(columns='origin_datetime')  # not needed
        )
        
        # To datetime
        forecast_weather['datetime'] = (
            pd.to_datetime(forecast_weather['datetime'])
            .dt.tz_localize(None)
        )

        # Add county
        forecast_weather[self.lat_lon_columns] = forecast_weather[self.lat_lon_columns].astype(float).round(1)
        forecast_weather = forecast_weather.merge(location, how='left', on=self.lat_lon_columns)
        
        # Modify column names - specify suffix
        forecast_weather = self.create_new_column_names(
            forecast_weather,
            suffix='_f',
            columns_no_change=self.lat_lon_columns + self.weather_join
        ) 
        
        # Group by & calculate aggregate stats 
        agg_columns = [col for col in forecast_weather.columns if col not in self.lat_lon_columns + self.weather_join]
        agg_dict = {agg_col: self.agg_stats for agg_col in agg_columns}
        forecast_weather = forecast_weather.groupby(self.weather_join).agg(agg_dict).reset_index() 
        
        # Flatten the multi column aggregates
        forecast_weather = self.flatten_multi_index_columns(forecast_weather)     
        return forecast_weather

    def create_electricity_features(self, electricity: pd.DataFrame) -> pd.DataFrame:
        """Create electricity prices features."""
        # To datetime
        electricity['forecast_date'] = pd.to_datetime(electricity['forecast_date'])
        
        # Test set has 1 day offset
        electricity['datetime'] = electricity['forecast_date'] + pd.DateOffset(1)
        
        # Modify column names - specify suffix
        electricity = self.create_new_column_names(
            electricity, 
            suffix='_electricity',
            columns_no_change=self.electricity_join
        )             
        return electricity

    def create_gas_features(self, gas: pd.DataFrame) -> pd.DataFrame:
        """Create gas prices features."""
        # Mean gas price
        gas['mean_price_per_mwh'] = (gas['lowest_price_per_mwh'] + gas['highest_price_per_mwh']) / 2
        
        # Modify column names - specify suffix
        gas = self.create_new_column_names(
            gas, 
            suffix='_gas',
            columns_no_change=self.gas_join
        )       
        return gas
    
    def process_features(self, data: pd.DataFrame, client: pd.DataFrame, 
                        historical_weather: pd.DataFrame, forecast_weather: pd.DataFrame, 
                        electricity: pd.DataFrame, gas: pd.DataFrame, 
                        location: pd.DataFrame) -> pd.DataFrame:
        """Process features from all datasets and merge together."""
        # Create features for relevant dataset
        data = self.create_data_features(data)
        client = self.create_client_features(client)
        historical_weather = self.create_historical_weather_features(historical_weather, location)
        forecast_weather = self.create_forecast_weather_features(forecast_weather, location)
        electricity = self.create_electricity_features(electricity)
        gas = self.create_gas_features(gas)
        
        # Merge all datasets into one df
        df = data.merge(client, how='left', on=self.client_join)
        df = df.merge(historical_weather, how='left', on=self.weather_join)
        df = df.merge(forecast_weather, how='left', on=self.weather_join)
        df = df.merge(electricity, how='left', on=self.electricity_join)
        df = df.merge(gas, how='left', on=self.gas_join)
        
        # Change columns to categorical for XGBoost
        df[self.category_columns] = df[self.category_columns].astype('category')
        return df


class TargetProcessor:
    """Process revealed targets for train and test sets."""
    
    @staticmethod
    def create_revealed_targets_train(data: pd.DataFrame, n_day_lags: int) -> pd.DataFrame:
        """Create past revealed_targets for train set based on number of day lags."""    
        original_datetime = data['datetime']
        revealed_targets = data[['datetime', 'prediction_unit_id', 'is_consumption', 'target']].copy()
        
        # Create revealed targets for all day lags
        for day_lag in range(2, n_day_lags + 1):
            revealed_targets['datetime'] = original_datetime + pd.DateOffset(day_lag)
            data = data.merge(
                revealed_targets, 
                how='left', 
                on=['datetime', 'prediction_unit_id', 'is_consumption'],
                suffixes=('', f'_{day_lag}_days_ago')
            )
        return data

    @staticmethod
    def create_revealed_targets_test(data: pd.DataFrame, previous_revealed_targets: List[pd.DataFrame], 
                                   n_day_lags: int) -> pd.DataFrame:
        """Create new test data based on previous_revealed_targets and N_day_lags.""" 
        for count, revealed_targets in enumerate(previous_revealed_targets):
            day_lag = count + 2
            
            # Get hour
            revealed_targets['hour'] = pd.to_datetime(revealed_targets['datetime']).dt.hour
            
            # Select columns and rename target
            revealed_targets = revealed_targets[['hour', 'prediction_unit_id', 'is_consumption', 'target']]
            revealed_targets = revealed_targets.rename(columns={"target": f"target_{day_lag}_days_ago"})
            
            # Add past revealed targets
            data = pd.merge(
                data,
                revealed_targets,
                how='left',
                on=['hour', 'prediction_unit_id', 'is_consumption'],
            )
            
        # If revealed_target_columns not available, replace by nan
        all_revealed_columns = [f"target_{day_lag}_days_ago" for day_lag in range(2, n_day_lags + 1)]
        missing_columns = list(set(all_revealed_columns) - set(data.columns))
        data[missing_columns] = np.nan 
        
        return data