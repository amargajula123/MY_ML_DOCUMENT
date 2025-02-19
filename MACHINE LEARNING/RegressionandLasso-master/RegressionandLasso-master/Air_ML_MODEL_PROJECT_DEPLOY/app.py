import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


#"request" will help you to capture the "json format" data
#that is coming from the "Postman".
app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    #return 'Hellow world'
    return render_template('home.html')



@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
   # print(data)
    new_data=[list(data.values())]
    #print(data.values())
    out=model.predict(new_data)[0]
    
    return jsonify(out)



@app.route('/predict',methods=['POST'])
def predict():
    # when we submited data in "html" page 
    # how our python code(app.py)/model is retrive here 
    # means from help of request.form.values we get the data 
    # 
    # for retriving the data from "html page" we use 
    # below steps
    data=request.form.values() # here we will get all the values
    print("DATA : ",data)
    data1=[float(i) for i in data]
    final_features = [np.array(data1)]

    print("FROM HTML PAGE DATA =",final_features)

    out=model.predict(final_features)[0]
    print("FROM HTML PAGE to 'DATA' OUTPUR PREDICTED VALUE = ",out)
    
    return render_template("home.html",prediction_text="Airfoil pressure is {}".format(out))


if __name__=="__main__":
    app.run(debug=True)