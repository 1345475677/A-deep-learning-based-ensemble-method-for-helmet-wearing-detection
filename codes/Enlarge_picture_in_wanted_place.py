import cv2 as cv
import pandas as pd
import numpy as np
import os
from codes.hat_person_cnn import test_single_img


imgdir="JPEGImages"
imgpath="JPEGImages/000019.jpg"
csvfile="data/test_picture_part2.csv"
save_csv="data/test_result_face_expand_judged.csv"


def cv_imread(filePath):
    cv_img=cv.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img
def cv_imwrite(save_Path,img):
    cv.imencode('.jpg', img)[1].tofile(save_Path)


def extract_single_wanted_image(img,list):
    time=1
   # print(list.loc['xmin'],list.loc['ymin'],list.loc['xmax'],list.loc['ymax'])
    dstimg=img[int(list.loc['ymin']):int(list.loc['ymax']),int(list.loc['xmin']):int(list.loc['xmax'])]
    dstimg=cv.resize(dstimg,(int(time*(list.loc['xmax']-list.loc['xmin'])),int(time*(list.loc['ymax']-list.loc['ymin']))),interpolation=cv.INTER_CUBIC)
    return dstimg


def get_small_pic(csv_file,imgdir):
    dir_to_save = "hat"
    dir_to_save2 = "person"
    file=pd.read_csv(csv_file,encoding='gbk')
    df=pd.DataFrame(file)
    hat_num=0
    person_num=0
    for i in range(len(csv_file)):
        imgpath=imgdir+'/'+df.loc[i,'filename']
        img=cv_imread(imgpath)
        dstimg=extract_single_wanted_image(img,df.loc[i])
        if(df.loc[i,'class']=='hat'):
            cv_imwrite(dir_to_save+'/'+str(hat_num)+'.jpg',dstimg)
            hat_num+=1
        elif(df.loc[i,'class']=='person'):
            cv_imwrite(dir_to_save2+'/'+str(person_num)+'.jpg',dstimg)
            person_num+=1
        if i==10:
            break



def resize_wanted_image(img,list,size):
    dstimg = img[int(list.loc['ymin']):int(list.loc['ymax']), int(list.loc['xmin']):int(list.loc['xmax'])]
    dstimg=cv.resize(dstimg,(size[0],size[1]))
    return dstimg

def class_judge(df):
    #file=pd.read_csv(csvfile,encoding='gbk')
    #df=pd.DataFrame(file)
    df.insert(loc=1,column='class',value=np.NAN)
    df['score']=0
    filename = df.loc[0, 'filename']
    img = cv_imread(imgdir + '/' + filename)
    for i in range(len(df)):
        if (filename != df.loc[i, 'filename']):
            filename = df.loc[i, 'filename']
            if os.path.isfile(imgdir + '/' + filename)==True:
                img = cv_imread(imgdir + '/' + filename)
            else:
                break
        dst=extract_single_wanted_image(img,df.loc[i])
        label,score=test_single_img(dst)
        print(label)
        if(label==0):
           df.loc[i,'class']="hat"
        elif(label==1):
           df.loc[i,'class']="person"
        else:
           print("i=",i,"label=",label)
        df.loc[i,'score']=score
    print(df)
    df.to_csv(save_csv,index=False)


def class_judge_video(video_path,df):
    df.insert(loc=1, column='class', value=np.NAN)
    df['score'] = 0
    print(df)
    video = cv.VideoCapture(video_path)
    success, img = video.read()
    k=0
    while(success and video.isOpened() and k<len(df)):
        dst = extract_single_wanted_image(img, df.loc[k])
        label, score = test_single_img(dst)
        print('score=',score)
        if (label == 0):
            df.loc[k, 'class'] = "hat"
        elif (label == 1):
            df.loc[k, 'class'] = "person"
        else:
            print("i=", k, "label=", label)
        df.loc[k,'score'] = score
        if(k<len(df)-1):
            if(df.loc[k+1,'frame']>df.loc[k,'frame']):
                success, img = video.read()
        k += 1
    video.release()

    return df


if __name__=="__main__":
    get_small_pic('JPEGImages')