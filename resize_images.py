from sklearn.utils import shuffle
import pandas as pd
import os
import numpy as np
from shutil import copyfile
import cv2



def resize(imagedir,csv_path):
    file=pd.read_csv(csv_path,encoding='gbk')
    df=pd.DataFrame(file)
    imagelist = os.listdir(imagedir)
    for filename in imagelist:
        image_path = os.path.join(imagedir, filename)
        img = cv2.imread(image_path)
        shape1=img.shape
        img=cv2.resize(img,(648,648))
        cv2.imwrite(image_path,img)
    for i in range(len(df)):
        scale1=float(648/df.loc[i,'height'])
        scale2=float(648/df.loc[i,'width'])
        df.loc[i,'xmin']*=scale2
        df.loc[i,'xmax']*=scale2
        df.loc[i,'ymin']*=scale1
        df.loc[i,'ymax']*=scale1
        df.loc[i,'height']=648
        df.loc[i,'width']=648
    df.to_csv(csv_path,index=False)






#train val
train_imagedir='helmet_5_test_data\data1/train/JPEGImages/'
train_csv_path='helmet_5_test_data\data1/train/helmet_train1.csv'

val_imagedir='helmet_5_test_data\data1/val/JPEGImages/'
val_csv_path='helmet_5_test_data\data1/val/helmet_val1.csv'


if __name__ == '__main__':
    resize(train_imagedir,train_csv_path)
    resize(val_imagedir,val_csv_path)
