# -*- coding: utf-8 -*- 

import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request 
app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def index():
    return render_template('website_main.html')

def predict(l):
    l=np.array([l])
    y_pred = model.predict(l)
    if y_pred > 0.5:
        return 1
    else:
        return 0
def predict_file(dataset):
    X=dataset.iloc[:,0:14]
    y_pred=model.predict(X)
    y_pred=(y_pred > 0.5)
    col=[]
    X=X.values
    for i,j in X[:,[0,1]]:
        if i==0 and j==0:
            col.append("California")
        elif i==1 and j==0:
            col.append("Colorado")
        else:
            col.append("Texas")
    X=pd.DataFrame(X)
    X=pd.DataFrame(X.drop([0,1], axis=1).values) 
    X.insert(0,"State",col,True)
    X.columns=['State','Age','Gender','Education','Marital Status', 'Employment Status','Income','Feedback','Assured Sum','Policy Beneficiaries','Premium','Health data']
    X["Churn"]=y_pred
    print(X.columns)
    return X.to_html()

@app.route('/churn_analysis')
def churn_analysis():
    with open('data_large.csv') as data_log:
        data_lines = data_log.readlines()
        dataset_sample = []
        rows = 20
        for x in range(rows):
            dataset_sample.append(data_lines[x].split(","))
    return render_template('churn_analysis.html', rows=rows, dataset_sample=dataset_sample)
       
@app.route('/risk_classifier')
def risk_classifier():
    return render_template('risk.html')

@app.route('/file_result', methods = ['POST'])
def file_result():
    if request.method == 'POST': 
        f=request.files['file']
        f.save(f.filename)
        dataset = pd.read_csv(f.filename,header=None, delimiter=',')
        res = predict_file(dataset)
        return render_template("file_result.html",result=res)
   
@app.route('/result', methods = ['POST']) 
def result():
     if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values())
        to_predict_list=[int(i) for i in to_predict_list]
        if to_predict_list[0] == 0:
            predict_list = [0,0]
        elif to_predict_list[0] == 1:
            predict_list = [1,0]
        else:
            predict_list = [0,1]
        for i in range(1,8):
            predict_list.append(to_predict_list[i])
        predict_list.append(to_predict_list[12])
        predict_list.append(to_predict_list[13])
        predict_list.append(to_predict_list[14])
        predict_list.append(to_predict_list[8]+to_predict_list[9]+to_predict_list[10]+to_predict_list[11])
        print(predict_list)
        y = predict(predict_list)
        y=int(y)
        return render_template("result.html", prediction =y, values=to_predict_list)


if __name__ == '__main__':
    app.run(debug=True)