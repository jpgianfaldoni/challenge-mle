import xgboost as xgb
import pickle
import numpy as np
from datetime import datetime

class XGBoostModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """Load the model from the pickle file."""
        try:
            with open(self.model_path, "rb") as file:
                self.model = pickle.load(file)
            print("Model loaded successfully.")
        except FileNotFoundError:
            raise Exception(f"Model file not found at {self.model_path}")

    def predict(self, input_data):
        """
        Make predictions using the loaded model.
        
        Args:
            input_data (list): List of lists (2D array) representing input features.
        Returns:
            list: Predictions for the input data.
        """
        if self.model is None:
            raise Exception("Model is not loaded. Call load_model() first.")
        dmatrix = xgb.DMatrix(np.array(input_data))
        predictions = self.model.predict(dmatrix)
        return predictions.tolist()
    
    def process_input(self, data):
        '''{
        'Store': number,
        'Dept': string,
        'IsHoliday': boolean,
        'Type': string,
        'Size':number,
        'date':timestamp,
        'MarkDown2': float,
        'MarkDown3': float,
        'MarkDown4': float,
        'MarkDown5': float,
        'CPI': float,
        'Unemployment': float
        }'''
        parsed_date = datetime.strptime(data['date'], "%d/%m/%Y")

        data['week'] = parsed_date.dt.isocalendar().week  
        data['month'] = parsed_date.month          
        data['day'] = parsed_date.day               

        timestamp = parsed_date.timestamp()  
        data_list = [
            data['Store'],
            data['Dept'], 
            bool(data['IsHoliday']),  
            data['Type'],  
            data['Size'],
            timestamp,  
            data['week'],
            data['month'],
            data['day'],
            data['MarkDown2'],
            data['MarkDown3'],
            data['MarkDown4'],
            data['MarkDown5'],
            data['CPI'],
            data['Unemployment']
        ]
        return
        