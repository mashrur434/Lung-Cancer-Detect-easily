
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

model = pickle.load(open('GaussianModel.pkl', 'rb'))

@app.route('/')
def home():
    return jsonify({"message": "API is running. Use /predict with POST method."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        gender_map = {"M": 1, "F": 0}
        GENDER = gender_map.get(data.get('GENDER'))

        AGE = float(data.get('AGE'))
        SMOKING = int(data.get('SMOKING'))
        YELLOW_FINGERS = int(data.get('YELLOW_FINGERS'))
        ANXIETY = int(data.get('ANXIETY'))
        PEER_PRESSURE = int(data.get('PEER_PRESSURE'))
        CHRONIC_DISEASE = int(data.get('CHRONIC_DISEASE'))
        FATIGUE = int(data.get('FATIGUE'))
        ALLERGY = int(data.get('ALLERGY'))
        WHEEZING = int(data.get('WHEEZING'))
        ALCOHOL_CONSUMING = int(data.get('ALCOHOL_CONSUMING'))
        COUGHING = int(data.get('COUGHING'))
        SHORTNESS_OF_BREATH = int(data.get('SHORTNESS_OF_BREATH'))
        SWALLOWING_DIFFICULTY = int(data.get('SWALLOWING_DIFFICULTY'))
        CHEST_PAIN = int(data.get('CHEST_PAIN'))

        features = pd.DataFrame([[
            GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE,
            CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING,
            COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN
        ]], columns=[
            'GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE',
            'CHRONIC DISEASE','FATIGUE ','ALLERGY ','WHEEZING',
            'ALCOHOL CONSUMING','COUGHING','SHORTNESS OF BREATH',
            'SWALLOWING DIFFICULTY','CHEST PAIN'
        ])

        prediction = model.predict(features)[0]

        result = {
            "prediction_raw": int(prediction),
            "prediction_text": "Yes (Lung Cancer patient)" if prediction == 1 else "No (No Lung cancer detected)"
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
