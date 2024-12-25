import pandas as pd
from prophet.serialize import model_from_json



class ProphetModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.mode = None

    def load_model(self):
        try:
            with open(self.model_path, 'r') as fin:
                self.model = model_from_json(fin.read()) 
            print("Model loaded successfully.")
        except FileNotFoundError:
            raise Exception(f"Model file not found at {self.model_path}")
        
    
    def create_future_df(self, df, store, dept, horizon):
        single_store = df[(df['Store'] == store) & (df['Dept'] == dept)]
        single_store = single_store.sort_values(by='ds', ascending=True)
        last_row = single_store.tail(1)
        last_date = pd.to_datetime(last_row['ds'].iloc[0])
        future_df = pd.DataFrame({
            'ds': [last_date + pd.DateOffset(weeks=i) for i in range(horizon)]
        })
        future_df = future_df.join(single_store.tail(horizon).reset_index(drop=True).drop(columns=['ds']))
        return future_df
    
    def load_and_process_data(self):
        df = pd.read_csv('data/combined_data.csv')
        df = df.sort_index()
        df.set_index('Date', inplace=True)
        df['lag_weekly_sales'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(52)
        df.reset_index(inplace=True)
        df = df.dropna(subset=['lag_weekly_sales'])
        store_data = df[['Date', 'Weekly_Sales', 'Size','Store', 'Dept','MarkDown1','MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'week', 'month', 'day', 'lag_weekly_sales']].copy()
        store_data.columns = ['ds', 'y', 'Size', 'Store', 'Dept','MarkDown1','MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'week', 'month', 'day', 'lag_weekly_sales']
        markDowns = ['MarkDown1','MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        for markdown in markDowns:
            store_data[markdown] = store_data[markdown].fillna(value=store_data[markdown].mean()) 
        return store_data
        

    def predict(self, data):
        store = data['Store']
        dept = data['Dept']
        horizon = data['Horizon']
        original_data = self.load_and_process_data()
        future_df = self.create_future_df(original_data, store,dept, horizon)
        forecast = self.model.predict(future_df)
        prediction = forecast[['ds', 'yhat']].copy()
        prediction.columns = ['date', 'weekly_sales']
        prediction['Store'] = store
        prediction['Dept'] = dept
        return prediction.to_dict(orient='records')


