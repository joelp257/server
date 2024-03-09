from flask import Flask, jsonify, request
import json
import pandas as pd
from pycaret.classification import *
from flask_cors import CORS

app = Flask(__name__)


@app.route('/query_d', methods=['GET'])
def process_query():
    try:
        #data = json_value()
        data = request.get_json()
        preg = data.get('Pregnancies')
        glu = data.get('Glucose')
        blo = data.get('BloodPressure')
        ski = data.get('SkinThickness')
        ins = data.get('Insulin')
        bmi = data.get('BMI')
        dia = data.get('DiabetesPedigreeFunction')
        age = data.get('Age')
        out = data.get('Outcome')
        print(preg)

        col_names = pd.read_pickle('./col_names.pkl')
        print(col_names)
        predictions=[preg,glu,blo,ski,ins,bmi,dia,age,out]
        df=pd.DataFrame([predictions],columns=col_names)
        model_loaded=load_model('./best_model')
        predicted = predict_model(model_loaded,data=df)
        prediction_list = predicted['prediction_label'].tolist()

        response_data = {           
            'result': 'success',
            'prediction_label':prediction_list
        }

        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
