from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import RepeatedKFold

filename=".csv"
file=pd.read_csv(filename,encoding='gbk')
df=pd.DataFrame(file)
df=df.fillna(value=0)
'''
file2=pd.read_csv(filename2,encoding='gbk')
df4=pd.DataFrame(file2)
df4=df4.fillna(value=0)
'''
df3=copy.deepcopy(df)


df2=pd.DataFrame(columns=df.columns)
k=0

for i in range(len(df)):
    if(df.loc[i,'y']==0):
        df2.loc[k,'filename']=df.loc[i,'filename']
        df2.loc[k,'class']=df.loc[i,'class']
        df2.loc[k,'xmin']=df.loc[i,'xmin']
        df2.loc[k,'ymin']=df.loc[i,'ymin']
        df2.loc[k,'xmax']=df.loc[i,'xmax']
        df2.loc[k,'ymax']=df.loc[i,'ymax']
        df2.loc[k,'area']=df.loc[i,'area']
        df2.loc[k,'score']=df.loc[i,'score']
        df2.loc[k,'score2']=df.loc[i,'score2']
        df2.loc[k,'y']=df.loc[i,'y']
        k+=1
print(type(df2.loc[0,'y']))
print(type(df.loc[0,'y']))
print(len(df2))

for i in range(0):
    df=df.append(df2,ignore_index=True)



print(df)

forecast_out=2100
adaboost=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2, min_samples_split=2, min_samples_leaf=1),
                            n_estimators=1000,
                            learning_rate=0.3,
                            algorithm='SAMME',
                            random_state=None
                            )


X=np.array(df[['area','score','score2']])
y=np.array(df['y'])
'''
X2=np.array(df4[['area','score','score2']])
y2=np.array(df4['y'])
'''
X3=np.array(df3[['area','score','score2']])
y3=np.array(df3['y'])

X_test=X[:len(X)-forecast_out]
y_test=y[:len(y)-forecast_out]
X_train=X[-forecast_out:]
y_train=y[-forecast_out:]
print(len(X_test))
print(len(X3))
adaboost.fit(X_train,y_train)
#adaboost=joblib.load('model.m')
score=adaboost.score(X3.astype(np.float64),y3.astype(np.float64))
score1=adaboost.score(X_train.astype(np.float64),y_train.astype(np.float64))
score2=adaboost.score(X_test.astype(np.float64),y_test.astype(np.float64))
#joblib.dump(adaboost,'model.m')

print(score)
print(score1)
print(score2)

label=adaboost.predict(X3)
sum=0
for i in range(len(label)):
    if(label[i]==df.loc[i,'y']):
        sum+=1
print(sum/len(label))


for i in range(len(label)):
    if label[i]==0:
        df=df.drop(i)
df.to_csv('test_result_final.csv',index=False)