from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from array import *
import sklearn

dfd = pd.read_csv("dataset/details .csv")
df=pd.read_csv("dataset/labeled.csv")

app = Flask(__name__)
model = pickle.load(open('scholar.pkl','rb'))

@app.route("/")
def home():
    return render_template("index.html")


scholarship=['INSPIRE Scholarship', 'National Fellowship Disabilities', 'Indira Gandhi Scholarship', 'Abdul Kalam Fellowship', 'AAI Sports Scholarship', 'Glow and lovely Scholarship', 'Dr. Ambedkar Scholarship', 'National Overseas Scholarship', 'Pragati Scholarship', 'ONGC Sports Scholarship']

sc = StandardScaler()

X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values
sc=StandardScaler()
X=sc.fit_transform(X)

@app.route("/", methods=['POST'])
def predict():
    if request.method == 'POST':
        Education = int(request.form['Education'])
        Gender = int(request.form['gender'])
        Community = int(request.form['Community'])
        Religion = int(request.form['Religion'])
        Exservice = int(request.form['Exservice'])
        Disability = int(request.form['Disability'])
        Sports = int(request.form['Sports'])
        Percentage = int(request.form['Percentage'])
        Income = int(request.form['Income'])
        India = int(request.form['India'])
        
        values=[Education, Gender, Community, Religion, Exservice, Disability,Sports, Percentage, Income,India]
        
        
        arr=[]
        for i in range(len(scholarship)):
            col = []
            col.append(i)
            for j in values:
                col.append(j)
            arr.append(col)
            
        eligible_scholarship =[]
        o=[]
        
        for i in range(len(scholarship)):
            val = sc.transform([arr[i]])
            output = model.predict(val).item()
            if(output>0.7):
                eligible_scholarship.append(scholarship[i])
                o.append(i)
                
        sch = []
        data = []
        
        for i in o:
            for j in range(1,6):
                    sch.append(dfd.iat[i,j])
            data.append(sch)
            sch = []
    
        return render_template("predict.html",data = data) 
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)