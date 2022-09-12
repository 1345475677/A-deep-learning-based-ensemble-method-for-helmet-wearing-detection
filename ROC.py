from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale

from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']



def ROC1(df):
    X = np.array(df[['score']])
    y = np.array(df['y'])
    f_num=0
    t_num=0
    FPR=[]
    TPR=[]
    for i in range(len(df)):
        if(X[i]==0):
            continue
        if(y[i]==0):
            f_num+=1
        else:
            t_num+=1
    print(f_num)
    print(t_num)
    for k in range(0,1001):
        thresh=k*0.001
        T=0
        F=0
        sum=0
        for i in range(len(df)):
            if(X[i]==0):
                continue
            if(X[i]>thresh):
                sum+=1
                if(y[i]==1):
                    T+=1
                elif(y[i]==0):
                    F+=1
        tpr=T/t_num
        fpr=F/f_num
        if(fpr<0.3):
            print(tpr,fpr)
        TPR.append(tpr)
        FPR.append(fpr)
    return TPR,FPR

def ROC2(df):
    X = np.array(df[['score2']])
    y = np.array(df['y'])
    f_num=0
    t_num=0
    FPR=[]
    TPR=[]
    for i in range(len(df)):
        if(X[i]==0):
            continue
        if(y[i]==0):
            f_num+=1
        else:
            t_num+=1
    for k in range(0,1001):
        thresh=k*0.001
        T=0
        F=0
        sum=0
        for i in range(len(df)):
            if(X[i]==0):
                continue
            if(X[i]>thresh):
                sum+=1
                if(y[i]==1):
                    T+=1
                elif(y[i]==0):
                    F+=1
        tpr=T/t_num
        fpr=F/f_num
        TPR.append(tpr)
        FPR.append(fpr)
    return TPR,FPR

def ROC3(y_score2,y):
    f_num = 0
    t_num = 0
    FPR = []
    TPR = []
    for i in range(len(y)):
        if (y[i] == 0):
            f_num += 1
        else:
            t_num += 1
    for k in range(0,1001):
        thresh=k*0.001
        T=0
        F=0
        for i in range(len(y)):
            if(y_score2[i][1]>thresh):
                if(y[i]==1):
                    T+=1
                elif(y[i]==0):
                    F+=1
        tpr=T/t_num
        fpr=F/f_num
        TPR.append(tpr)
        FPR.append(fpr)
    return TPR,FPR


def PR(df,sum=1651):
    pass


def show_adaboost(tpr,fpr,savepath):
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    #plt.scatter(fpr,tpr)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('fpr',fontdict={'size':16})
    plt.ylabel('tpr',fontdict={'size':16})
    plt.title('adaboost ROC')
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    plt.show()

def show_object(tpr,fpr,savepath):
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    #plt.scatter(fpr,tpr)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('fpr',fontdict={'size':16})
    plt.ylabel('tpr',fontdict={'size':16})
    plt.title('Faster RCNN ROC')
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    plt.show()

def show_Tiny(tpr,fpr,savepath):
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    #plt.scatter(fpr,tpr)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('fpr',fontdict={'size':16})
    plt.ylabel('tpr',fontdict={'size':16})
    plt.title('Tiny face+CNN ROC')
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    plt.show()


def show2(tpr,fpr,tpr1,fpr1,tpr2,fpr2):
    roc_auc = auc(fpr, tpr)
    roc_auc1=auc(fpr1,tpr1)
    roc_auc2=auc(fpr2,tpr2)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='adaboost_auc (area = %0.2f)' % roc_auc)
    plt.plot(fpr1, tpr1, color='k',
             lw=lw, label='rcnn_auc (area = %0.2f)' % roc_auc1)
    plt.plot(fpr2, tpr2, color='g',
             lw=lw, label='tiny_face_auc (area = %0.2f)' % roc_auc2)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('fpr',fontdict={'size':16})
    plt.ylabel('tpr',fontdict={'size':16})
    plt.title('three kinds of ROC')
    plt.legend(loc="lower right")
    plt.savefig("data/adaboost_rcnn_tiny_face_ROC.png")
    plt.show()



filename="data/test_result_final1.csv"
file=pd.read_csv(filename,encoding='gbk')
df=pd.DataFrame(file)
df=df.fillna(value=0)
adaboost=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2, min_samples_split=2, min_samples_leaf=1),
                            n_estimators=1000,
                            learning_rate=0.3,
                            algorithm='SAMME',
                            random_state=None
                            )

X=np.array(df[['area','score','score2']])
y=np.array(df['y'])
#X=scale(X)
kf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)

sum=0
k=0
for train_index, test_index in kf.split(X,y):
    adaboost.fit(X[train_index].astype(np.float64),y[train_index].astype(np.float64))
    score=adaboost.score(X[test_index].astype(np.float64),y[test_index].astype(np.float64))
    print(score)
    sum+=score
    if(k==5):
        joblib.dump(adaboost,"codes/adaboost_model/model.m")
        y_score=adaboost.decision_function(X[test_index])
        y_score2=adaboost.predict_proba(X[test_index])
        # print(y_score)
        # print(y_score2)
        #fpr, tpr, threshold = roc_curve(y[test_index], y_score)
        #print(fpr,tpr)
        tpr,fpr=ROC3(y_score2,y[test_index])
        show_adaboost(tpr=tpr,fpr=fpr,savepath="ROCIMG/adaboost.jpg")
        tpr1,fpr1=ROC1(df)
        #print(fpr,'\n',tpr)
        show_object(tpr1,fpr1,"ROCIMG/object_detection.jpg")
        tpr2,fpr2=ROC2(df)
        show_Tiny(tpr2,fpr2,"ROCIMG/Tiny_face.jpg")
        show2(tpr, fpr, tpr1, fpr1, tpr2, fpr2)
    k+=1

print("ave=",sum/10)

