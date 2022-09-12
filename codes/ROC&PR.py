from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import roc_curve, auc,average_precision_score
import matplotlib.pyplot as plt


def show_PRfig(y,x,savepath):
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


def show_ROCfig(y,x,savepath):
    roc_auc = auc(x, y)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    #plt.scatter(fpr,tpr)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False positive',fontdict={'size':16})
    plt.ylabel('True positive',fontdict={'size':16})
    # plt.title('Faster RCNN ROC')
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    plt.show()


def show_allPR3(y,x,y1,x1,y2,x2,savepath):
    roc_auc = auc(x, y)
    roc_auc1=auc(x1,y1)
    roc_auc2=auc(x2,y2)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, color='darkorange',
             lw=lw, label='Mobilenet(area = %0.2f)' % roc_auc)
    plt.plot(x1, y1, color='k',
             lw=lw, label='Tiny Face+CNN (area = %0.2f)' % roc_auc1)
    plt.plot(x2, y2, color='g',
             lw=lw, label='Mobilenet+Tiny Face+CNN (area = %0.2f)' % roc_auc2)

    # plt.scatter(fpr,tpr)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('recall', fontdict={'size': 16})
    plt.ylabel('precision', fontdict={'size': 16})
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    plt.show()

def show_allPR4(y,x,y1,x1,y2,x2,y3,x3,savepath):
    roc_auc = auc(x, y)
    roc_auc1=auc(x1,y1)
    roc_auc2=auc(x2,y2)
    roc_auc3=auc(x3,y3)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, color='darkorange',
             lw=lw, label='Faster_RCNN (area = %0.3f)' % roc_auc)
    plt.plot(x1, y1, color='k',
             lw=lw, label='Tiny_Face (area = %0.3f)' % roc_auc1)
    plt.plot(x2, y2, color='g',
             lw=lw, label='ensemble (area = %0.3f)' % roc_auc2)
    plt.plot(x3, y3, color='r',
             lw=lw, label='mobile (area = %0.3f)' % roc_auc3)

    # plt.scatter(fpr,tpr)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('recall', fontdict={'size': 16})
    plt.ylabel('precision', fontdict={'size': 16})
    # plt.title('Faster RCNN ROC')
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    plt.show()

def show_allROC3(y,x,y1,x1,y2,x2,savepath):
    roc_auc = auc(x, y)
    roc_auc1 = auc(x1, y1)
    roc_auc2 = auc(x2, y2)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, color='darkorange',
             lw=lw, label='Mobilenet (area = %0.2f)' % roc_auc)
    plt.plot(x1, y1, color='k',
             lw=lw, label='Tiny Face+CNN (area = %0.2f)' % roc_auc1)
    plt.plot(x2, y2, color='g',
             lw=lw, label='Mobilenet+Tiny Face+CNN (area = %0.2f)' % roc_auc2)

    # plt.scatter(fpr,tpr)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False positive', fontdict={'size': 16})
    plt.ylabel('True positive', fontdict={'size': 16})
    # plt.title('Faster RCNN ROC')
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    plt.show()
def show_allROC4(y,x,y1,x1,y2,x2,y3,x3,savepath):
    roc_auc = auc(x, y)
    roc_auc1 = auc(x1, y1)
    roc_auc2 = auc(x2, y2)
    roc_auc3 = auc(x3, y3)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, color='darkorange',
             lw=lw, label='Faster_RCNN (area = %0.3f)' % roc_auc)
    plt.plot(x1, y1, color='k',
             lw=lw, label='Tiny_Face (area = %0.3f)' % roc_auc1)
    plt.plot(x2, y2, color='g',
             lw=lw, label='ensemble (area = %0.3f)' % roc_auc2)
    plt.plot(x3, y3, color='r',
             lw=lw, label='mobile (area = %0.3f)' % roc_auc3)

    # plt.scatter(fpr,tpr)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False positive', fontdict={'size': 16})
    plt.ylabel('True positive', fontdict={'size': 16})
    # plt.title('Faster RCNN ROC')
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    plt.show()

def get_rate(X,y):
    # threshs=copy.copy(X)
    # qsort(threshs,0,len(X)-1)
    FPR = []
    TPR = []
    FNR = []
    TNR =[]
    PRECISION=[]
    RECALL=[]
    # for thresh in threshs:
    t_num=0
    f_num=0
    for i in range(len(y)):
        if X[i]>0:
            if y[i]==0:
                f_num+=1
            elif y[i]==1:
                t_num+=1

    for k in range(0, 1001):
        thresh = k * 0.001
        tp=0
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(len(X)):
            if (X[i] < thresh and y[i] == 1):
                FN += 1
            elif(X[i] < thresh and y[i] == 0):
                if(X[i]>0):
                    TN+=1
            if (X[i] >= thresh and X[i]>0):
                if (y[i] == 1):
                    TP += 1
                elif (y[i] == 0):
                    FP += 1

        if(TP+FP==0):
            precision=1
        else:
            precision=float(TP)/float(TP+FP)
        if(TP+FN==0 or k==0):
            recall=1
        else:
            recall = float(TP) / float(TP + FN)#recall
        if (t_num==0 or k==0):
            tpr = 1
        else:
            tpr = float(TP) / float(t_num)  # recall
        if(FP+TN==0 ):
            tnr=1
        else:
            tnr = float(TN) / float(FP + TN)
        if(TP+FN==0):
            fnr=1
        else:
            fnr = float(FN) / float(TP + FN)
        if(f_num==0 or k==0):
            fpr=1
        else:
            fpr = float(FP) / float(f_num)
        TPR.append(tpr)
        FPR.append(fpr)
        FNR.append(fnr)
        TNR.append(tnr)
        RECALL.append(recall)
        PRECISION.append(precision)

        if(k==1):
            print(TP,TP+FN)

    return TPR, FPR, FNR,TNR,PRECISION,RECALL

def PR(df):
    adaboost = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=2, min_samples_split=2, min_samples_leaf=1),
        n_estimators=1000,
        learning_rate=0.03,
        algorithm='SAMME.R',
        random_state=None
        )
    X = np.array(df[['area', 'score', 'score2']])
    X1 = np.array(df['score'])
    X2=np.array(df['score2'])
    y = np.array(df['tg'])
    #Faster RCNN
    # TPR1, FPR1, FNR1, TNR1,precision1,recall1=get_rate(X1,y)

    # show_ROCfig(TPR1, FPR1, savepath='.png')
    # FPR1, TPR1, threshold = roc_curve(y, X1)
    # precision1, recall1, thresholds = precision_recall_curve(y, X1)
    # show_PRfig(precision1, recall1,savepath='.png')
    # show_ROCfig(tpr,fpr,savepath='.png')

    #Tiny Face

    # TPR2, FPR2, FNR2, TNR2,precision2,recall2=get_rate(X2,y)

    # precision2, recall2, thresholds = precision_recall_curve(y, X2)
    # FPR2, TPR2, threshold = roc_curve(y, X2)
    # show_PRfig(precision, recall, savepath='.png')
    # show_ROCfig(TPR,FPR,savepath='.png')
    #ensemble
    kf = RepeatedKFold(n_splits=2, n_repeats=1, random_state=0)
    sum = 0
    k = 0
    for train_index, test_index in kf.split(X, y):

        # Faster RCNN
        TPR1, FPR1, FNR1, TNR1, precision1, recall1 = get_rate(X1[test_index], y[test_index])
        TPR2, FPR2, FNR2, TNR2, precision2, recall2 = get_rate(X2[test_index], y[test_index])


        k+=1
        adaboost.fit(X[train_index].astype(np.float64), y[train_index].astype(np.float64))
        score = adaboost.score(X[test_index].astype(np.float64), y[test_index].astype(np.float64))
        print(score)
        y_score=adaboost.decision_function(X[test_index])
        y_score2=adaboost.predict_proba(X[test_index])
        S=[]
        for s in y_score2:
           S.append(s[1])
        y_score2=S
        print(X[test_index].shape)
        print(len(y_score2))
        temp=0
        temp1=0
        for i in range(len(y_score2)):
            if X[test_index][i][1]==0 and X[test_index][i][2]==0:
                y_score2[i]=0
                y_score[i]=0


        TPR3, FPR3, FNR3,TNR3, precision3, recall3 = get_rate(y_score, y[test_index])

        # precision3, recall3, thresholds = precision_recall_curve(y[test_index], y_score2)
        # FPR3, TPR3, threshold = roc_curve(y[test_index], y_score2)

        if(k==1):
            scorelist = []
            for i in range(len(TPR1)):
                scorelist.append(
                    [recall1[i], precision1[i], TPR1[i], FPR1[i], recall2[i], precision2[i], TPR2[i], FPR2[i],
                     recall3[i], precision3[i], TPR3[i], FPR3[i]])
            scoredf = pd.DataFrame(scorelist,
                                   columns=['recall1', 'precision1', 'TPR1', 'FPR1', 'recall2', 'precision2', 'TPR2',
                                            'FPR2', 'recall3', 'precision3', 'TPR3', 'FPR3'])
            scoredf.to_csv('scorelist.csv')

        show_allPR3(precision1,recall1,precision2,recall2,precision3,recall3,savepath='Figs/all_PR3.png')
        show_allROC3(TPR1,FPR1,TPR2,FPR2,TPR3,FPR3,savepath='Figs/all_ROC3.png')


def ave(df1,df2,df3,df4,df5):
    dflist=[df1,df2,df3,df4,df5]
    TPR1=[];FPR1=[];FNR1=[];TNR1=[];PRECISION1=[];RECALL1=[]
    TPR2 = [];FPR2 = [];FNR2 = [];TNR2 = [];PRECISION2 = [];RECALL2=[]
    TPR3 = [];FPR3 = [];FNR3 = [];TNR3 = [];PRECISION3 = [];RECALL3=[]
    MAP1=[];MAP2=[];MAP3=[]
    for df in dflist:
        df = df.fillna(value=0)
        X = np.array(df[['area', 'score', 'score2']])
        X1 = np.array(df['score'])
        X2 = np.array(df['score2'])
        y = np.array(df['tg'])
        ''''''
        map1=average_precision_score(y,X1)
        map2=average_precision_score(y,X2)
        tpr1, fpr1, fnr1, tnr1, precision1,recall1 = get_rate(X1, y)
        tpr2, fpr2, fnr2, tnr2, precision2,recall2 = get_rate(X2, y)
        # ensemble
        kf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=0)
        tpr3=[]
        fpr3=[]
        fnr3=[]
        tnr3=[]
        precision3=[]
        recall3=[]
        map3=[]
        for train_index, test_index in kf.split(X, y):
            adaboost = AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=2, min_samples_split=2, min_samples_leaf=1),
                n_estimators=1000,
                learning_rate=0.3,
                algorithm='SAMME',
                random_state=None
            )
            adaboost.fit(X[train_index].astype(np.float64), y[train_index].astype(np.float64))
            y_score = adaboost.decision_function(X[test_index])
            y_score2 = adaboost.predict_proba(X[test_index])
            S = []
            for s in y_score2:
                S.append(s[1])
            y_score2 = S
            for i in range(len(y_score2)):
                if X[test_index][i][1] == 0 and X[test_index][i][2] == 0:
                    y_score2[i] = 0
                    y_score[i]=0
            mape=average_precision_score(y[test_index],y_score)

            tpre, fpre, fnre, tnre, precisione,recalle = get_rate( y_score, y[test_index])
            tpr3.append(tpre)
            fpr3.append(fpre)
            fnr3.append(fnre)
            tnr3.append(tnre)
            precision3.append(precisione)
            recall3.append(recalle)
            map3.append(mape)
        tpr4 = [];fpr4 = [];fnr4 = [];tnr4 = [];precision4 = [];recall4=[]
        for i in range(len(tpr3)):
            if (i == 0):
                tpr4 = tpr3[i]
                fpr4 = fpr3[i]
                fnr4 = fnr3[i]
                tnr4 = tnr3[i]
                precision4 = precision3[i]
                recall4=recall3[i]
            else:
                tpr4 = list(np.array(tpr3[i]) + np.array(tpr4))
                fpr4 = list(np.array(fpr3[i]) + np.array(fpr4))
                fnr4 = list(np.array(fnr3[i]) + np.array(fnr4))
                tnr4 = list(np.array(tnr3[i]) + np.array(tnr4))
                precision4 = list(np.array(precision3[i]) + np.array(precision4))
                recall4=list(np.array(recall3[i])+np.array(recall4))
        tpr3 = np.array(tpr4) / len(tpr3)
        fpr3 = np.array(fpr4) / len(fpr3)
        fnr3 = np.array(fnr4) / len(fnr3)
        tnr3 = np.array(tnr4) / len(tnr3)
        precision3 = np.array(precision4) / len(precision3)
        recall3=np.array(recall4)/len(recall3)
        map3=sum(map3)/len(map3)

        TPR1.append(tpr1)
        FPR1.append(fpr1)
        FNR1.append(fnr1)
        TNR1.append(tnr1)
        PRECISION1.append(precision1)
        RECALL1.append(recall1)
        MAP1.append(map1)

        TPR2.append(tpr2)
        FPR2.append(fpr2)
        FNR2.append(fnr2)
        TNR2.append(tnr2)
        PRECISION2.append(precision2)
        RECALL2.append(recall2)
        MAP2.append(map2)

        TPR3.append(tpr3)
        FPR3.append(fpr3)
        FNR3.append(fnr3)
        TNR3.append(tnr3)
        PRECISION3.append(precision3)
        RECALL3.append(recall3)
        MAP3.append(map3)

    TPR=[];FPR=[];FNR=[];TNR=[];PRECISION=[];RECALL=[]
    for i in range(len(TPR1)):
        if(i==0):
            TPR=TPR1[i]
            FPR=FPR1[i]
            FNR=FNR1[i]
            TNR=TNR1[i]
            PRECISION=PRECISION1[i]
            RECALL=RECALL1[i]
        else:
            TPR=list(np.array(TPR1[i])+np.array(TPR))
            FPR=list(np.array(FPR1[i])+np.array(FPR))
            FNR=list(np.array(FNR1[i])+np.array(FNR))
            TNR=list(np.array(TNR1[i])+np.array(TNR))
            PRECISION=list(np.array(PRECISION1[i])+np.array(PRECISION))
            RECALL=list(np.array(RECALL1[i])+np.array(RECALL))
    TPR1=np.array(TPR)/len(TPR1)
    FPR1=np.array(FPR)/len(FPR1)
    FNR1=np.array(FNR)/len(FNR1)
    TNR1=np.array(TNR)/len(TNR1)
    PRECISION1=np.array(PRECISION)/len(PRECISION1)
    RECALL1=np.array(RECALL)/len(RECALL1)
    #tinyface
    for i in range(len(TPR2)):
        if (i == 0):
            TPR = TPR2[i]
            FPR = FPR2[i]
            FNR = FNR2[i]
            TNR = TNR2[i]
            PRECISION = PRECISION2[i]
            RECALL=RECALL2[i]
        else:
            TPR = list(np.array(TPR2[i]) + np.array(TPR))
            FPR = list(np.array(FPR2[i]) + np.array(FPR))
            FNR = list(np.array(FNR2[i]) + np.array(FNR))
            TNR = list(np.array(TNR2[i]) + np.array(TNR))
            PRECISION = list(np.array(PRECISION2[i]) + np.array(PRECISION))
            RECALL=list(np.array(RECALL2[i])+np.array(RECALL))
    TPR2 = np.array(TPR) / len(TPR2)
    FPR2 = np.array(FPR) / len(FPR2)
    FNR2 = np.array(FNR) / len(FNR2)
    TNR2 = np.array(TNR) / len(TNR2)
    PRECISION2 = np.array(PRECISION) / len(PRECISION2)
    RECALL2=np.array(RECALL)/len(RECALL2)
        # ensenmble
    for i in range(len(TPR3)):
        if (i == 0):
            TPR = TPR3[i]
            FPR = FPR3[i]
            FNR = FNR3[i]
            TNR = TNR3[i]
            PRECISION = PRECISION3[i]
            RECALL=RECALL3[i]
        else:
            TPR = list(np.array(TPR3[i]) + np.array(TPR))
            FPR = list(np.array(FPR3[i]) + np.array(FPR))
            FNR = list(np.array(FNR3[i]) + np.array(FNR))
            TNR = list(np.array(TNR3[i]) + np.array(TNR))
            PRECISION = list(np.array(PRECISION3[i]) + np.array(PRECISION))
            RECALL=list(np.array(RECALL3[i])+np.array(RECALL))
    TPR3 = np.array(TPR) / len(TPR3)
    FPR3 = np.array(FPR) / len(FPR3)
    FNR3 = np.array(FNR) / len(FNR3)
    TNR3 = np.array(TNR) / len(TNR3)
    PRECISION3 = np.array(PRECISION) / len(PRECISION3)
    RECALL3=np.array(RECALL)/len(RECALL3)

    MAP1=sum(MAP1)/len(MAP1)
    MAP2=sum(MAP2)/len(MAP2)
    MAP3=sum(MAP3)/len(MAP3)

    scorelist=[]
    for i in range(len(TPR1)):
        scorelist.append([RECALL1[i],PRECISION1[i],TPR1[i],FPR1[i],RECALL2[i],PRECISION2[i],TPR2[i],FPR2[i],RECALL3[i],PRECISION3[i],TPR3[i],FPR3[i]])
    scoredf=pd.DataFrame(scorelist,columns=['recall1','precision1','TPR1','FPR1','recall2','precision2','TPR2','FPR2','recall3','precision3','TPR3','FPR3'])
    scoredf.to_csv('scorelist.csv')

    return TPR1,FPR1,RECALL1,PRECISION1,TPR2,FPR2,RECALL2,PRECISION2,TPR3,FPR3,RECALL3,PRECISION3

def show4figs():
    filename1 = "helmet_5_test_data\data1/test/test_result_final0.1.csv"
    filename2 = "helmet_5_test_data\data2/test/test_result_final0.1.csv"
    filename3 = "helmet_5_test_data\data3/test/test_result_final0.1.csv"
    filename4 = "helmet_5_test_data\data4/test/test_result_final0.1.csv"
    filename5 = "helmet_5_test_data\data5/test/test_result_final0.1.csv"
    filename6 = "helmet_5_test_data\data1/test/test_result_darknet.csv"
    file1 = pd.read_csv(filename1, encoding='gbk')
    file2 = pd.read_csv(filename2, encoding='gbk')
    file3 = pd.read_csv(filename3, encoding='gbk')
    file4 = pd.read_csv(filename4, encoding='gbk')
    file5 = pd.read_csv(filename5, encoding='gbk')
    file6=pd.read_csv(filename6,encoding='gbk')
    df1 = pd.DataFrame(file1)
    df2 = pd.DataFrame(file2)
    df3 = pd.DataFrame(file3)
    df4 = pd.DataFrame(file4)
    df5 = pd.DataFrame(file5)
    df6=pd.DataFrame(file6)
    df6=df6.fillna(value=0)
    X4 = np.array(df6['score'])
    y4 = np.array(df6['tg'])
    TPR1,FPR1,RECALL1,PRECISION1,TPR2,FPR2,RECALL2,PRECISION2,TPR3,FPR3,RECALL3,PRECISION3=ave(df1, df2, df3, df4, df5)
    TPR4, FPR4, FNR4, TNR4, PRECISION4, RECALL4 = get_rate(X4, y4)
    show_allPR4(PRECISION1,RECALL1,PRECISION2,RECALL2,PRECISION3,RECALL3,PRECISION4,RECALL4,savepath='helmet_5_test_data/all_PR.png')
    show_allROC4(TPR1,FPR1,TPR2,FPR2,TPR3,FPR3,TPR4,FPR4,savepath='helmet_5_test_data/all_ROC.png')



def show3figs():
    filename1 = "helmet_5_test_data\data1/test/test_result_final0.1.csv"
    filename2 = "helmet_5_test_data\data2/test/test_result_final0.1.csv"
    filename3 = "helmet_5_test_data\data3/test/test_result_final0.1.csv"
    filename4 = "helmet_5_test_data\data4/test/test_result_final0.1.csv"
    filename5 = "helmet_5_test_data\data5/test/test_result_final0.1.csv"
    file1 = pd.read_csv(filename1, encoding='gbk')
    file2 = pd.read_csv(filename2, encoding='gbk')
    file3 = pd.read_csv(filename3, encoding='gbk')
    file4 = pd.read_csv(filename4, encoding='gbk')
    file5 = pd.read_csv(filename5, encoding='gbk')
    df1 = pd.DataFrame(file1)
    df2 = pd.DataFrame(file2)
    df3 = pd.DataFrame(file3)
    df4 = pd.DataFrame(file4)
    df5 = pd.DataFrame(file5)
    TPR1, FPR1, RECALL1, PRECISION1, TPR2, FPR2, RECALL2, PRECISION2, TPR3, FPR3, RECALL3, PRECISION3 = ave(df1, df2,
                                                                                                            df3, df4,
                                                                                                            df5)

    show_allPR3(PRECISION1, RECALL1, PRECISION2, RECALL2, PRECISION3, RECALL3,
                savepath='helmet_5_test_data/all_PR3.png')
    show_allROC3(TPR1, FPR1, TPR2, FPR2, TPR3, FPR3, savepath='helmet_5_test_data/all_ROC3.png')

def main():
    filename1 = "helmet_5_test_data\data1/test/test_result_final0.1.csv"
    filename2 = "helmet_5_test_data\data2/test/test_result_final0.1.csv"
    filename3 = "helmet_5_test_data\data3/test/test_result_final0.1.csv"
    filename4 = "helmet_5_test_data\data4/test/test_result_final0.1.csv"
    filename5 = "helmet_5_test_data\data5/test/test_result_final0.1.csv"
    file1=pd.read_csv(filename1,encoding='gbk')
    file2=pd.read_csv(filename2,encoding='gbk')
    file3=pd.read_csv(filename3,encoding='gbk')
    file4=pd.read_csv(filename4,encoding='gbk')
    file5=pd.read_csv(filename5,encoding='gbk')
    df1=pd.DataFrame(file1)
    df2=pd.DataFrame(file2)
    df3=pd.DataFrame(file3)
    df4=pd.DataFrame(file4)
    df5=pd.DataFrame(file5)
    ave(df1,df2,df3,df4,df5)



def main_darknet():
    filename = "helmet_5_test_data\data1/test/test_result_mobile.csv"
    file = pd.read_csv(filename, encoding='gbk')
    df = pd.DataFrame(file)
    df = df.fillna(value=0)
    x0=0
    sum1=0
    for i in range(len(df)):
        if(df.loc[i,'tg']==1):
            sum1+=1
        if(df.loc[i,'score']>0 and df.loc[i,'tg']==1):
            x0+=1
    print(x0,sum1)


    X1 = np.array(df['score'])
    y = np.array(df['tg'])
    tpr1, fpr1, fnr1, tnr1, precision1, recall1 = get_rate(X1, y)
    # fpr1, tpr1, threshold = roc_curve(y, X1)
    # precision1, recall1, thresholds = precision_recall_curve(y, X1)
    show_PRfig(precision1, recall1,savepath='helmet_5_test_data\data1/test\Figs/mobilie0.25_PR.png')
    show_ROCfig(tpr1,fpr1,savepath='helmet_5_test_data\data1/test\Figs/mobile0.25_ROC.png')







if __name__ == '__main__':
    # show4figs()
    filename = "D:\学习\helmet_5_test_data\data1/test/test_result_final_mobile.csv"
    file = pd.read_csv(filename, encoding='gbk')
    df = pd.DataFrame(file)
    df = df.fillna(value=0)
    PR(df)
    # show3figs()


# if __name__ == '__main__':
#     main()
#     #0.90，63.6
#     #