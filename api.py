from flask import Flask, jsonify,request
import json
import pandas as pd
from pycaret.classification import *
from pycaret.regression import *

app = Flask(__name__)


@app.route('/diabetes', methods=['GET'])
def process_diabetic_query():
    try:
        data = request.get_json()
        glu = data.get('Glucose')
        blo = data.get('BloodPressure')
        ski = data.get('SkinThickness')
        ins = data.get('Insulin')
        bmi = data.get('BMI')
        dia = data.get('DiabetesPedigreeFunction')
        age = data.get('Age')

        col_names = pd.read_pickle('.\d_pred_col_names.pkl')
        print(col_names)
        predictions=[glu,blo,ski,ins,bmi,dia,age,1]
        df=pd.DataFrame([predictions],columns=col_names)
        model_loaded=load_model('.\d_pred.pickle')
        predicted = predict_model(model_loaded,data=df)

        response_data = {
            'result': 'success',
            'prediction_label': predicted['prediction_label']
        }

        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/cholestrol', methods=['GET'])
def process_chol_query():
    try:
        data = request.get_json()
        age = data.get('age')
        sex = data.get('sex')
        cp = data.get('cp')
        trest = data.get('trestbps')
        cho = data.get('chol')

        col_names = pd.read_pickle('\chol_pred_col_names.pkl')
        print(col_names)
        predictions=[age,sex,cp,trest,0]
        df=pd.DataFrame([predictions],columns=col_names)
        model_loaded=load_model('\chol_pred.pickle')
        predicted = predict_model(model_loaded,data=df)

        response_data = {
            'result': 'success',
            'prediction_label': predicted['prediction_label']
        }

        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/bp', methods=['GET'])
def process_bp_query():
    try:
        data = request.get_json()
        glu = data.get('Glucose')
        blo = data.get('BloodPressure')
        ski = data.get('SkinThickness')
        ins = data.get('Insulin')
        bmi = data.get('BMI')
        dia = data.get('DiabetesPedigreeFunction')
        age = data.get('Age')

        col_names = pd.read_pickle('.\bp_pred_col_names.pkl')
        print(col_names)
        predictions=[glu,0,ski,ins,bmi,dia,age]
        df=pd.DataFrame([predictions],columns=col_names)
        model_loaded=load_model('.\bp_pred.pickle')
        predicted = predict_model(model_loaded,data=df)

        response_data = {
            'result': 'success',
            'prediction_label': predicted['prediction_label']
        }

        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
