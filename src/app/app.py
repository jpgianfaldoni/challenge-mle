from flask import Flask, request, jsonify
from model_handler import XGBoostModel
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_next_4_weeks_xgboost():
    '''
    Input example: input = {
            'Store': 1,
            'Dept': 1,
            'Type':'A'.
            'Size': 1234,
            'date': '26/02/2011'
        }
    '''
    try:
        data = request.get_json()
        start_date = datetime.strptime(data['date'], '%d/%m/%Y')
        predictions = []
        for i in range(4):
            input = {
            'Store': data['Store'],
            'Dept': data['Dept'],
            'date': start_date.strftime('%d/%m/%Y'),
            'Size':data['Size'],
            'Type':data['Type']
            }
            predictions.append(model.predict(input))
        return jsonify({"prediction": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    model_path = 'src/models/4_weeks_prediction_xgboost.pkl'
    preprocessor_path = 'src/models/4_weeks_prediction_preprocessor.pkl'
    model = XGBoostModel(model_path,preprocessor_path)
    model.load_model()
    model.load_preprocessor()
    app.run(debug=True)
