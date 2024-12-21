# challenge-mle

## Dependencies
To install the dependencies run `pip install -r requirements.txt` on the root folder

## Project structure
### Notebooks
Folder containing the following notebooks:

data_exploring.ipynb: initial exploration and insights discoveries on the provided data

data_processing.ipynb: processing data to be used as input for the models

4_weeks_model_training.ipynb: training the XGBoost model that predicts the sales for each department 4 weeks ahead

year_prediction_model_training.ipynb: training the XGBoost model that predicts the sales for each department 1 year ahead

prophet.ipynb: training the Prophet model that predicts the sales for each department 1 year ahead

next_year_predictions.ipynb: creating the predictions for each store/department for the next year

### Data
Original and processed data

### src
#### /app
app.py: flask api

model_handler.py: Model class and methods
#### /models
Treined models and preprocessors

