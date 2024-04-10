from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

application = Flask(__name__)
app = application

regrassor = pickle.load(open('E:\Diabetes Prediction Project\model\\regrassor.pkl','rb'))
scaler = pickle.load(open('E:\Diabetes Prediction Project\model\scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():

    result = ""

    if request.method=='POST':
        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = int(request.form.get('Age'))

        new_scaler = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        data = regrassor.predict(new_scaler)

        if data[0]==1:
            result='Diabetic'
        else:
            result='Non Diabetic'

        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')

if __name__=='__main__':
    app.run(host='0.0.0.0')