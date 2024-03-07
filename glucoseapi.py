from flask import Flask, jsonify
import json
import pandas as pd
from pycaret.classification import *

app = Flask(__name__)

def json_value():
    input_data = {
        "Pregnancies": 6,
        "Glucose": 159,
        "BloodPressure": 82,
        "SkinThickness": 25,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
        "Outcome": 1,
    }
    #json_payload = json.dumps(input_data)
    #return json_payload
    return input_data

@app.route('/query', methods=['GET'])
def process_query():
    try:
        data = json_value()
        preg = data.get('Pregnancies')
        glu = data.get('Glucose')
        blo = data.get('BloodPressure')
        ski = data.get('SkinThickness')
        ins = data.get('Insulin')
        bmi = data.get('BMI')
        dia = data.get('DiabetesPedigreeFunction')
        age = data.get('Age')
        out = data.get('Outcome')
        print(glu)

        glucol_names = pd.read_pickle('D:/col_names.pkl')
        print(glucol_names)
        predictions=[preg,glu,blo,ski,ins,bmi,dia,age,out]
        df=pd.DataFrame([predictions],columns=glucol_names)
        model_loaded=load_model('D:/best_model')
        predicted = predict_model(model_loaded,data=df)
        prediction_list = predicted['prediction_label'].tolist()

        response_data = {
            'result': 'success',
            'prediction_label': prediction_list
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
