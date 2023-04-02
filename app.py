
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from array import *
import sklearn


df=pd.read_csv("scholar_data.csv")


app = Flask(__name__)
model = pickle.load(open('scholar.pkl','rb'))

@app.route("/",methods=['GET'])
def home():
    return render_template("index.html")


scholarship=['INSPIRE Scholarship', 'National Fellowship Disabilities', 'Indira Gandhi Scholarship', 'Abdul Kalam Fellowship', 'AAI Sports Scholarship', 'Glow and lovely Scholarship', 'Dr. Ambedkar Scholarship', 'National Overseas Scholarship', 'Pragati Scholarship', 'ONGC Sports Scholarship']
qualification_map={"Undergraduate":0,"Postgraduate":1,"Doctrate":2}
gender_map = {"Male":1,"Female":0}
community_map = {"SC/ST":0,"OBC":1,"General":2,"Minority":3}
annual_percentage_map = {"90-100":0,"80-90":1,"70-80":2,"60-70":3}
income_map = {"Upto 1.5L":0,"1.5L to 3L":1,"3L to 6L":2,"Above 6L":3}
india_map= {"In":0,"Out":1}
religion_map = {"Hindu":0,"Muslim":1,"Chirstian":2,"Others":3}
exservice_map={"Yes":1,"No":0}
disability_map={"Yes":1,"No":0}
sports_map={"Yes":1,"No":0}
scholarship_map = {'INSPIRE Scholarship': 0,'National Fellowship Disabilities':1,'Indira Gandhi Scholarship':2,'Abdul Kalam Fellowship':3,'AAI Sports Scholarship':4,'Glow and lovely Scholarship':5,'Dr. Ambedkar Scholarship':6,'National Overseas Scholarship':7,'Pragati Scholarship':8,'ONGC Sports Scholarship':9}


df2 = df.copy()

df2["Education Qualification"]=df2["Education Qualification"].map(qualification_map)
df2["Gender"]=df2["Gender"].map(gender_map)
df2["Community"]=df2["Community"].map(community_map)
df2["Annual-Percentage"]=df2["Annual-Percentage"].map(annual_percentage_map)
df2["Income"]=df2["Income"].map(income_map)
df2["Religion"]=df2["Religion"].map(religion_map)
df2["India"]=df2["India"].map(india_map)
df2["Exservice-men"]=df2["Exservice-men"].map(exservice_map)
df2["Disability"]=df2["Disability"].map(disability_map)
df2["Sports"]=df2["Sports"].map(sports_map)
df2["Name"]=df["Name"].map(scholarship_map)


sc = StandardScaler()

X = df2[df2.columns[:-1]].values
y = df2[df2.columns[-1]].values
sc=StandardScaler()
X=sc.fit_transform(X)

@app.route("/predict", methods=['POST'])
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

        for i in range(len(scholarship)):
            val = sc.transform([arr[i]])
            output = model.predict(val).item()
            if(output>0):
                eligible_scholarship.append(scholarship[i])
        
        return render_template("predict.html",eligible_scholarship=eligible_scholarship,length=len(eligible_scholarship))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)