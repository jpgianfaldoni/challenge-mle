from flask import Flask, request, jsonify
from model_handler import XGBoostModel

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_next_4_weeks():
    '''
    Input example: input_obj = {
            'store_data': [
                {
                    'Store': 1,
                    'Dept': 1,
                    'IsHoliday': True,
                    'Type': 'A',
                    'Size': 321312312312,
                    'date': '26/02/2011'
                },
                        {
                    'Store': 12,
                    'Dept': 12,
                    'IsHoliday': True,
                    'Type': 'B',
                    'Size': 1234,
                    'date': '05/03/2011'
                },
                        {
                    'Store': 1,
                    'Dept': 1,
                    'IsHoliday': True,
                    'Type': 'A',
                    'Size': 151315,
                    'date': '12/03/2011'
                },
                        {
                    'Store': 1,
                    'Dept': 1,
                    'IsHoliday': True,
                    'Type': 'A',
                    'Size': 151315,
                    'date': '19/03/2011'
                },
            ]
        }
    '''
    try:
        data = request.get_json()
        predictions = []
        for store in data['store_data']:
            predictions.append(model.predict(store))
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
