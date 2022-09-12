import os
import xml.dom.minidom
import cv2 as cv
import numpy as np
import pandas as pd



# imgpath="JPEGImages"
imgpath='image'
csvfile=".csv"
savepath="data/test"

def cv_imread(filePath):
    cv_img=cv.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img
def cv_imwrite(save_Path,img):
    cv.imencode('.jpg', img)[1].tofile(save_Path)

def draw_rectangle(imgpath,csvfile,savepath):
    k=0
    file=pd.read_csv(csvfile,encoding='gbk')
    df=pd.DataFrame(file)
    filename=df.loc[0,'filename']
    img=cv_imread(imgpath+'/'+filename)
    for i in range(len(df)):
        if(filename!=df.loc[i,'filename']):
            cv_imwrite(savepath+'/'+filename,img)
            filename=df.loc[i,'filename']
            img=cv_imread(imgpath+'/'+filename)
            if(df.loc[i,'class']!="person" and df.loc[i,'class']!='hat'):
                continue
        xmin=int(df.loc[i,'xmin'])
        xmax=int(df.loc[i,'xmax'])
        ymin=int(df.loc[i,'ymin'])
        ymax=int(df.loc[i,'ymax'])
        objectname=df.loc[i,'class']
        if(objectname=='person'):
            cv.rectangle(img,(xmin, ymin), (xmax, ymax), (0, 0, 255),thickness=2)
            cv.putText(img, "not_wear",(xmin, ymin), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255),thickness=2)
        elif(objectname=='hat'):
            cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
            cv.putText(img, "wear_helmet", (xmin, ymin), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), thickness=2)
def expand_face_rectangle_in_csv(csvfile):
    proportion=0.15
    file=pd.read_csv(csvfile,encoding='gbk')
    df=pd.DataFrame(file)
    for i in range(len(df)):
        xmin=df.loc[i,'xmin']
        xmax=df.loc[i,'xmax']
        ymin=df.loc[i,'ymin']
        ymax=df.loc[i,'ymax']
        xgap=xmax-xmin
        ygap=ymax-ymin
        xmin=xmin-xgap*proportion*0.5
        ymin=ymin-ygap*proportion*0.5
        xmax=xmax+xgap*proportion*0.5
        ymax=ymax+ygap*proportion*0.5
        if xmin<=0:
            xmin=1
        if ymin<=0:
            ymin=1
        df.loc[i,'xmin']=xmin
        df.loc[i,'xmax']=xmax
        df.loc[i,'ymin']=ymin
        df.loc[i,'ymax']=ymax
    savepath2=csvfile.split('.')[0]+"_expand.csv"
    df.to_csv(savepath2)




if __name__=="__main__":
    draw_rectangle(imgpath,csvfile,savepath)







