# challenge-mle

## How to Run

Clone the Repository

```
git clone https://github.com/your-username/challenge-mle.git
cd challenge-mle
```

Install dependencies
`brew install libomp`
`pip install -r requirements.txt`

Run the Application

`python src/app/app.py`

Make a POST request to the /predict endpoint, this is a body example to predict the next 4 weeks:

```
{
    'store_data': [
        {
            'Store': 1,
            'Dept': 1,
            'IsHoliday': True,
            'Type': 'A',
            'Size': 1234,
            'date': '26/02/2011'
        },
                {
            'Store': 1,
            'Dept': 1,
            'IsHoliday': False,
            'Type': 'A',
            'Size': 12345,
            'date': '05/03/2011'
        },
                {
            'Store': 1,
            'Dept': 1,
            'IsHoliday': False,
            'Type': 'A',
            'Size': 123456,
            'date': '12/03/2011'
        },
                {
            'Store': 1,
            'Dept': 1,
            'IsHoliday': True,
            'Type': 'A',
            'Size': 123456,
            'date': '19/03/2011'
        },
    ]
}
```

## Project structure

### Presentation

Folder containing the final presentation notebook **final_presentation.ipynb**

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
