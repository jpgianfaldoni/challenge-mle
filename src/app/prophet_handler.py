import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from pathlib import Path
from prophet.diagnostics import performance_metrics



class ProphetModel:
    def __init__(self):
        self.regressors = ['Size','Store', 'Dept','MarkDown1','MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        self.holidays = self.create_holidays()
        self.forecast = None
        self.model = None
    def load_store_data(self, store, department):
        base_path = Path(__file__).resolve().parent.parent.parent 
        data_path = base_path / 'data' / 'combined_data.csv'
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.fillna(0)
        store_data = df[(df['Store'] == store) & (df['Dept'] == department)]
        store_data = store_data[['Date', 'Weekly_Sales', 'Size','Store', 'Dept','MarkDown1','MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']]
        store_data.columns = ['ds', 'y', 'Size', 'Store', 'Dept','MarkDown1','MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'] 
        return store_data

    def create_holidays(self):
        new_years = pd.DataFrame({
        'holiday': 'new_year',
        'ds': pd.to_datetime(['2010-12-31', '2011-12-30']),
        })
        thanksgiving = pd.DataFrame({
        'holiday': 'thanksgiving',
        'ds': pd.to_datetime(['2010-11-26', '2011-11-25']),
        'lower_window': 0,
        'upper_window': 1,
        })
        superbowls = pd.DataFrame({
        'holiday': 'superbowl',
        'ds': pd.to_datetime(['2010-02-12', '2012-02-10', '2011-02-11']),
        'lower_window': 0,
        'upper_window': 1,
        })
        labor_days = pd.DataFrame({
        'holiday': 'labor_day',
        'ds': pd.to_datetime(['2010-09-10', '2011-09-09', '2012-09-07']),
        'lower_window': 0,
        'upper_window': 1,
        })
        return pd.concat((new_years, superbowls, thanksgiving,labor_days))
 
    def create_model(self):
        model = Prophet(holidays=self.holidays, holidays_prior_scale = 50)
        model.add_country_holidays(country_name='US')
        for regressor in self.regressors:
            model.add_regressor(regressor)
        return model

    def plot_prediction(self):
        self.model.plot(self.forecast)
        plt.title(f"Forecast for Store")
        plt.show()
    def predict(self, store, department):
        store_data = self.load_store_data(store, department)
        model = self.create_model()
        self.model = model
        model.fit(store_data)
        future = model.make_future_dataframe(periods=52, freq = 'W')
        for regressor in self.regressors:
            future[regressor] = store_data[regressor].iloc[-1]  
        forecast = model.predict(future)
        self.forecast = forecast
        next_year = forecast[['ds', 'yhat']].tail(52).copy()
        next_year.rename(columns={'ds': 'date', 'yhat': 'weekly_sales_prediction'}, inplace=True)
        next_year.loc[:, 'store'] = store
        next_year.loc[:, 'department'] = department
        return next_year

    def performance(self):
        from prophet.diagnostics import cross_validation
        df_cv = cross_validation(self.model, initial='365 days', period='15 days', horizon = '365 days')
        df_p = performance_metrics(df_cv)
        return df_p