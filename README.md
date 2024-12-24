# challenge-mle

## How to Run

Clone the Repository

```
git clone git@github.com:jpgianfaldoni/challenge-mle.git
cd challenge-mle
```

Install dependencies
`brew install libomp`
`pip install -r requirements.txt`

Run the Application

`python src/app/app.py`

Make a POST request to the /predict endpoint, this is a body example to predict the next 4 weeks:

```
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"Store": 1, "Dept": 1, "date": "26/02/2011", "Type": "A", "Size": 1234}'
```

## Project structure

### Presentation

Folder containing the presentation notebook **presentation.ipynb**

### Notebooks

Folder containing the following notebooks:

**data_exploring.ipynb**: initial exploration and insights discoveries on the provided data

**data_processing.ipynb**: processing data to be used as input for the models

**4_weeks_model_training.ipynb**: training the XGBoost model that predicts the sales for each department 4 weeks ahead

### Data

Original and processed data

### src

#### /app

**app.py**: flask api

**model_handler.py**: XGBoost model class

**prophet_handler.py** Prophet model class

#### /models

Treined models and preprocessors
