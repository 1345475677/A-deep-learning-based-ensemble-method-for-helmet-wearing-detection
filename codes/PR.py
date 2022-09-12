from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
from sklearn.externals import joblib

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
def area(list1, list2):
    xmax1 = list1.loc['xmax']
    xmin1 = list1.loc['xmin']
    ymin1 = list1.loc['ymin']
    ymax1 = list1.loc['ymax']
    xmax2 = list2.loc['xmax']
    xmin2 = list2.loc['xmin']
    ymin2 = list2.loc['ymin']
    ymax2 = list2.loc['ymax']
    if (xmax1 == np.NAN):
        return 0
    xmax3 = min(float(xmax1), float(xmax2))
    xmin3 = max(float(xmin1), float(xmin2))
    ymax3 = min(float(ymax1), float(ymax2))
    ymin3 = max(float(ymin1), float(ymin2))

    area1 = abs((xmax1 - xmin1) * (ymax1 - ymin1))
    area2 = abs((xmax2 - xmin1) * (ymax2 - ymin2))
    area3 = (xmax3 - xmin3) * (ymax3 - ymin3)
    min_area = min(area1, area2)
    if (xmax3 - xmin3 <= 0 or ymax3 - ymin3 <= 0):
        return 0
    return area3 / min_area

def merge(df1,df2):
    df2.insert(loc=len(df2.columns), column='exist', value=0)
    df1.insert(loc=len(df1.columns), column='tg', value=1)
    # df1.insert(loc=len(df1.columns), column='score', value=0)
    # df1.insert(loc=len(df1.columns), column='score2', value=0)
    df2 = df2.fillna(value=0)
    val = 0.4
    j1 = 0
    j2 = 0
    i = 0
    while (True):
        if j2 < len(df2):
            while (df2.loc[j1, 'filename'] == df2.loc[j2, 'filename']):
                j2 += 1
                if j2 == len(df2):
                    break
            while (df1.loc[i, 'filename'] > df2.loc[j1, 'filename']):
                j1 = j2
                while (df2.loc[j1, 'filename'] == df2.loc[j2, 'filename']):
                    j2 += 1
                    if j2 == len(df2):
                        break
            while (df1.loc[i, 'filename'] < df2.loc[j1, 'filename']):
                j2 = j1
                while (df2.loc[j1, 'filename'] == df2.loc[j2, 'filename']):
                    j1 -= 1
                    if (j1 == 0):
                        break
        flag = 0
        for j in range(j1, j2):
            dif3 = area(df1.loc[i], df2.loc[j])
            if dif3 > 1 - val and df1.loc[i, 'class'] == df2.loc[j, 'class']:
                flag = 1
                df1.loc[i,'score']=df2.loc[j,'score']
                df1.loc[i,'score2']=df2.loc[j,'score2']
                df2.loc[j, 'exist'] = 1
                # break
            elif dif3 > 1 - val and df1.loc[i, 'class'] != df2.loc[j, 'class']:
                if flag == 1 and df2.loc[j, 'exist'] == 1:
                    break
                else:
                    df2.loc[j, 'exist'] = 1
                    flag = 2
                    df1.loc[i, 'score'] = 1 - df2.loc[j, 'score']
                    df1.loc[i, 'score2'] = 1 - df2.loc[j, 'score2']
        if (i < len(df1) - 1):
            if (df1.loc[i, 'filename'] != df1.loc[i + 1, 'filename']):
                x = pd.DataFrame(columns=df1.columns)
                k = 0
                for j in range(j1, j2):
                    if (df2.loc[j, 'exist'] == 0):
                        df2.loc[j, 'exist'] = 1
                        x.loc[k, 'filename'] = df2.loc[j, 'filename']
                        x.loc[k, 'class'] = df2.loc[j, 'class']
                        x.loc[k, 'xmin'] = df2.loc[j, 'xmin']
                        x.loc[k, 'ymin'] = df2.loc[j, 'ymin']
                        x.loc[k, 'xmax'] = df2.loc[j, 'xmax']
                        x.loc[k, 'ymax'] = df2.loc[j, 'ymax']
                        x.loc[k, 'score'] = df2.loc[j, 'score']
                        x.loc[k, 'score2'] = df2.loc[j, 'score2']
                        k += 1
                if (len(x) > 0):
                    above = df1.loc[:i]
                    below = df1.loc[i + 1:]
                    df1 = above.append(x, ignore_index=True).append(below, ignore_index=True)
                    while (df1.loc[i, 'filename'] == df1.loc[i + 1, 'filename']):
                        i += 1
        elif (i == len(df1) - 1):
            x = pd.DataFrame(columns=df1.columns)
            k = 0
            for j in range(j1, j2):
                if (df2.loc[j, 'exist'] == 0):
                    df2.loc[j, 'exist'] = 1
                    x.loc[k, 'filename'] = df2.loc[j, 'filename']
                    x.loc[k, 'class'] = df2.loc[j, 'class']
                    x.loc[k, 'xmin'] = df2.loc[j, 'xmin']
                    x.loc[k, 'ymin'] = df2.loc[j, 'ymin']
                    x.loc[k, 'xmax'] = df2.loc[j, 'xmax']
                    x.loc[k, 'ymax'] = df2.loc[j, 'ymax']
                    x.loc[k, 'score'] = df2.loc[j, 'score']
                    x.loc[k, 'score2'] = df2.loc[j, 'score2']
                    k += 1
            if (len(x) > 0):
                above = df1.loc[:i]
                below = df1.loc[i + 1:]
                df1 = above.append(x, ignore_index=True).append(below, ignore_index=True)
                i = len(df1) - 1
        if i < len(df1):
            i += 1
        if i == len(df1):
            break
    return df1

def Mergefile():
    filename1="helmet_test.csv"
    filename2 = "test_result_final1.csv"
    file1=pd.read_csv(filename1,encoding='gbk')
    file2=pd.read_csv(filename2,encoding='gbk')
    df1=pd.DataFrame(file1)
    df2=pd.DataFrame(file2)
    print(df1)
    print(df2)
    df1=merge(df1,df2)
    print(df1)
    df1.to_csv("PR.csv")


def PR1(df):
    X = np.array(df[['score']])
    y = np.array(df['tg'])
    fp = 0
    tp = 0
    fn=0
    FPR = []
    TPR = []
    FNR = []
    for i in range(len(df)):
        if (X[i] == 0 and y[i]==1):
            fn+=1
        if (y[i] == 0):
            fp+= 1
        else:
            tp += 1
    print(fp)
    print(tp)
    print(fn)
    for k in range(0, 1001):
        thresh = k * 0.001
        T = 0
        F = 0
        FN=0
        sum = 0
        for i in range(len(df)):
            if (X[i] == 0 and y[i]==1):
                FN+=1
            if (X[i] > thresh):
                sum += 1
                if (y[i] == 1):
                    T += 1
                elif (y[i] == 0):
                    F += 1
        if(T+F==0):
            tpr = 1
            fpr = 1
        else:
            tpr = T / (T+F)
            fpr = F / (T+F)
        fnr = T/ (FN+T)
        TPR.append(tpr)
        FPR.append(fpr)
        FNR.append(fnr)
    return TPR,FPR,FNR

def show_fig(y,x,savepath):
    roc_auc = auc(x, y)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, color='darkorange',
             lw=lw, label='PR curve (area = %0.3f)' % roc_auc)
    #plt.scatter(fpr,tpr)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('recall',fontdict={'size':16})
    plt.ylabel('precison',fontdict={'size':16})
    # plt.title('Faster RCNN ROC')
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    plt.show()

if __name__ == '__main__':
    adaboost=joblib.load("model.m")
    filename="PR.csv"
    file=pd.read_csv(filename,encoding='gbk')
    df=pd.DataFrame(file)
    for i in range(len(df)):
        df.loc[i,'area']=(df.loc[i,'xmax']-df.loc[i,'xmin'])*(df.loc[i,'ymax']-df.loc[i,'ymin'])
    df = df.fillna(value=0)
    X = np.array(df[['area','score','score2']])
    y = np.array(df['tg'])
    X1=np.array(df[['score2']])
    print(average_precision_score(y, X1))
    precision, recall, thresholds = precision_recall_curve(y, X1)
    show_fig(precision, recall,savepath="PR/Tiny_Face.PNG")
    y_score=adaboost.decision_function(X)
    print(average_precision_score(y, y_score))
    precision, recall, thresholds = precision_recall_curve(y, y_score)
    show_fig(precision, recall,savepath="PR/ensemble.PNG")
