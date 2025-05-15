from flask import Flask,render_template,request,jsonify
import pickle 
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler


application=Flask(__name__)
app=application

regression_model  = pickle.load(open('models/regression.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_data():
    if request.method=="POST":
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = request.form['Classes']  # 'fire' or 'not fire'
        region = request.form['region']    # '0' or '1'

        Classes_encoded = 1 if Classes.strip().lower() == 'fire' else 0
        # Convert region to float (already numerical in form as '0' or '1')
        region_encoded = float(region)

        data=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes_encoded,region_encoded]])
        result = regression_model.predict(data)

        return render_template('prediction.html', prediction=result[0])
    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(debug=True)