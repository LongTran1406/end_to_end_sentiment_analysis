import pandas as pd

class DataManager:
    def __init__(self, config):
        self.config = config
    
    @staticmethod
    def read_data(path):
        return pd.read_csv(path)
    
    @staticmethod
    def save_data(df, path):
        return df.to_csv(path, index = False)
    