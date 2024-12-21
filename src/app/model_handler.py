import pickle
from datetime import timedelta,datetime
import pandas as pd


class XGBoostModel:
    def __init__(self, model_path,preprocessor_path):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None

    def load_model(self):
        try:
            with open(self.model_path, "rb") as file:
                self.model = pickle.load(file)
            print("Model loaded successfully.")
        except FileNotFoundError:
            raise Exception(f"Model file not found at {self.model_path}")
        

    def load_preprocessor(self):
        try:
            with open(self.preprocessor_path, "rb") as file:
                self.preprocessor = pickle.load(file)
            print("Preprocessor loaded successfully.")
        except FileNotFoundError:
            raise Exception(f"Preprocessor file not found at {self.preprocessor_path}")

    def predict(self, input_data):
        # '''{
        # 'Store': number,
        # 'Dept': string,
        # 'IsHoliday': boolean,
        # 'Type': string,
        # 'Size':number,
        # 'date':timestamp
        # }'''
        processed_data = self.process_input(input_data)
        input_data_transformed = self.preprocessor.transform(processed_data)
        predictions = self.model.predict(input_data_transformed)
        return predictions.tolist()
    

    def create_lagged_features(self,df, store, dept, date):
        df['Date'] = pd.to_datetime(df['Date'])
        date = pd.to_datetime(date)
        reference_date = date - timedelta(weeks=4)
        
        df_filtered = df[(df['Store'] == store) & (df['Dept'] == dept)]
        if df_filtered.empty:
            raise ValueError("No data found for the specified store and department.")
        
        shiftable_columns = ['Weekly_Sales', 'Temperature', 'MarkDown1', 'Fuel_Price',
                            'MarkDown2', 'MarkDown3', 'MarkDown4', 
                            'MarkDown5', 'CPI', 'Unemployment']
        
        lags = [4, 8, 16, 32]
        lagged_features = pd.DataFrame(index=[0])
        
        for lag in lags:
            for col in shiftable_columns:
                lagged_col_name = f'lag_{lag}_{col}'
                lagged_value = df_filtered[col].shift(lag).iloc[-1]
                lagged_features[lagged_col_name] = [lagged_value]
        
        week_number = reference_date.isocalendar()[1]  
        for col in shiftable_columns:
            avg_col_name = f'{col}_historical_week_avg'
            historical_avg = df_filtered[df_filtered['Date'].dt.isocalendar().week == week_number][col].mean()
            lagged_features[avg_col_name] = [historical_avg]
        return lagged_features
    
    def process_input(self, data):        
        parsed_date = datetime.strptime(data['date'], "%d/%m/%Y")

        data['week'] = parsed_date.isocalendar().week
        data['month'] = parsed_date.month
        data['day'] = parsed_date.day  
        original_data = {
            'Date': parsed_date,
            'Store': data['Store'],
            'Dept': data['Dept'], 
            'IsHoliday': bool(data['IsHoliday']),  
            'Type': data['Type'],  
            'Size': data['Size'],
            'week': data['week'],
            'month': data['month'],
            'day': data['day']
        }
        df = pd.read_csv('../../data/combined_data.csv')
        df = self.create_lagged_features(df, original_data['Store'],original_data['Dept'],original_data['Date'])
        for key, value in original_data.items():
            df[key] = value
        return df
