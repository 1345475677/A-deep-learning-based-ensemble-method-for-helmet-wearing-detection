from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import copy

filename="data/test_final1.csv"
modelpath='adaboost_model/model.m'

adaboost=joblib.load(modelpath)

file=pd.read_csv(filename,encoding='gbk')
df=pd.DataFrame(file)
df=df.fillna(value=0)
X=np.array(df[['area','score','score2']])

label=adaboost.predict(X)
label2=adaboost.predict_proba(X)
print(label)
print(label2)

print(len(label),len(df))


label3=[]
for i in range(len(label2)):
    label3.append(label2[i][1])
print(len(label3))


df['score3']=label3
print(df)

for i in range(len(label)):
    if label[i]==0:
        df=df.drop(i)


print(len(df))
print(df)
df.to_csv('data/test_final.csv',index=False)
