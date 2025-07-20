import torch

class Config:
    """Configuration class for the pipeline."""
    
    def __init__(self):
        self.data_dir = "data/"
        self.output_dir = "output/"
        self.model_dir = "models/"
        self.debug = False
        self.n_day_lags = 15
        self.n_estimators = 1500
        self.early_stopping_rounds = 100
        self.train_block_ids = list(range(0, 600))
        
        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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