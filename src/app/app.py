from flask import Flask, request, jsonify
from model_handler import XGBoostModel
from prophet_handler import ProphetModel
from datetime import datetime

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_next_4_weeks_xgboost():
    '''
    Input example: input = {
            'store': 1,
            'dept': 1,
            'type':'A',
            'size': 1234,
            'date': '26/02/2011'
        }
    '''
    try:
        data = request.get_json()
        start_date = datetime.strptime(data['date'], '%d/%m/%Y')
        predictions = []
        for i in range(4):
            input = {
            'Store': data['store'],
            'Dept': data['dept'],
            'date': start_date.strftime('%d/%m/%Y'),
            'Size':data['size'],
            'Type':data['type']
            }
            predictions.append(model_xgboost.predict(input))
        return jsonify({"prediction": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/predict-prophet", methods=["POST"])
def predict_prophet():
    '''
    Input example: input = {
            'store': 1,
            'dept': 1,
            'type':'A',
            'size': 1234,
            'date': '26/02/2011',
            'horizon' : 4
        }
    '''
    try:
        data = request.get_json()
        start_date = datetime.strptime(data['date'], '%d/%m/%Y')
        predictions = []
        input = {
        'Store': data['store'],
        'Dept': data['dept'],
        'date': start_date.strftime('%d/%m/%Y'),
        'Size':data['size'],
        'Type':data['type'],
        'Horizon':data['horizon']
        }
        predictions.append(model_prophet.predict(input))
        return jsonify({"prediction": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    xgboost_model_path = 'src/models/4_weeks_prediction_xgboost.pkl'
    prophet_model_path = 'src/models/prophet_1_year.json'
    preprocessor_path = 'src/models/4_weeks_prediction_preprocessor.pkl'
    model_xgboost = XGBoostModel(xgboost_model_path,preprocessor_path)
    model_xgboost.load_model()
    model_xgboost.load_preprocessor()
    model_prophet = ProphetModel(prophet_model_path)
    model_prophet.load_model()

    app.run(debug=True)
