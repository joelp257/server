from flask import Flask, jsonify,request
import json
import pandas as pd
from pycaret import classification
from pycaret  import regression

app = Flask(__name__)


@app.route('/')
def index():
    return 'Index Page'


@app.route('/test')
def process_test_query():
    return 'success'


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

        col_names = pd.read_pickle('./diabetes_prediction/d_pred_col_names.pkl')
        print(col_names)
        predictions=[glu,blo,ski,ins,bmi,dia,age,1]
        df=pd.DataFrame([predictions],columns=col_names)
        model_loaded=classification.load_model('./diabetes_prediction/d_pred')
        predicted = classification.predict_model(model_loaded,data=df)

        response_data = {
            'result': 'success',
            'prediction_label': predicted['prediction_label'].to_json()
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

        col_names = pd.read_pickle('./cholestreol_prediction/chol_pred_col_names.pkl')
        print(col_names)
        predictions=[age,sex,cp,trest,0]
        df=pd.DataFrame([predictions],columns=col_names)
        model_loaded= regression.load_model('./cholestreol_prediction/chol_pred')
        predicted = regression.predict_model(model_loaded,data=df)

        response_data = {
            'result': 'success',
            'prediction_label': predicted['prediction_label'].to_json()
        }

        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/bp', methods=['GET'])
def process_bp_query():
    try:
        data = request.get_json()
        glu = data.get('Glucose')
        ski = data.get('SkinThickness')
        ins = data.get('Insulin')
        bmi = data.get('BMI')
        dia = data.get('DiabetesPedigreeFunction')
        age = data.get('Age')

        col_names = pd.read_pickle('./bp_prediction/bp_pred_col_names.pkl')
        print(col_names)
        predictions=[glu,0,ski,ins,bmi,dia,age]
        df=pd.DataFrame([predictions],columns=col_names)
        model_loaded=regression.load_model('./bp_prediction/bp_pred')
        predicted = regression.predict_model(model_loaded,data=df)

        response_data = {
            'result': 'success',
            'prediction_label': predicted['prediction_label'].to_json()
        }

        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
