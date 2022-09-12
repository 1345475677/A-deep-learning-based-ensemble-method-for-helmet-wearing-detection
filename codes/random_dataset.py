from sklearn.utils import shuffle
import pandas as pd
import os
import numpy as np
from shutil import copyfile

def get_csv(dir_path,filename):
    if(os.path.isdir(dir_path)):
        list=os.listdir(dir_path)
        column_name = ['filename']
        df = pd.DataFrame(list, columns=column_name)
        df=random_dataset(df)
        print(df)
        df.to_csv(filename,index=False)


def random_dataset(df):
    df=shuffle(df)
    return df



def get_training_and_testing(csv_filename,data_dir):
    JPG_dir="helmet_5_test_data\JPEGImages"
    Ann_dir="helmet_5_test_data\Annotations"
    base_name="data"
    file=pd.read_csv(csv_filename,encoding='gbk')
    df=pd.DataFrame(file)
    for i in range(1,2):
        above=df.loc[:len(df)/5*(i-1)]
        below=df.loc[len(df)/5*i:]
        train_val = above.append(below, ignore_index=True)
        train_names=train_val.loc[len(train_val)/4:]
        train_names = np.array(train_names)
        val_names=train_val.loc[:len(train_val)/4]
        val_names=np.array(val_names)
        test_names=df.loc[len(df)/5*(i-1):len(df)/5*i]
        test_names=np.array(test_names)
        print(len(train_names),len(val_names),len(test_names))
        dir_name=data_dir+'/'+base_name+str(i)
        if(not os.path.exists(dir_name)):
            os.makedirs(dir_name)
        for filename in test_names:
            src1=JPG_dir
            dst1=dir_name+'/test'+'/'+'JPEGImages'
            src2=Ann_dir
            dst2=dir_name+'/test'+'/'+'Annotations'
            if (not os.path.exists(dst1)):
                os.makedirs(dst1)
            if (not os.path.exists(dst2)):
                os.makedirs(dst2)
            copyfile(src1+'/'+filename[0],dst1+'/'+filename[0])
            copyfile(src2 + '/' + filename[0].split('.')[0]+'.xml', dst2 + '/' + filename[0].split('.')[0]+'.xml')
        for filename in train_names:
            src1 = JPG_dir
            dst1 = dir_name + '/train' + '/' + 'JPEGImages'
            src2 = Ann_dir
            dst2 = dir_name + '/train' + '/' + 'Annotations'
            if (not os.path.exists(dst1)):
                os.makedirs(dst1)
            if (not os.path.exists(dst2)):
                os.makedirs(dst2)
            copyfile(src1+'/'+filename[0],dst1+'/'+filename[0])
            copyfile(src2 + '/' + filename[0].split('.')[0]+'.xml', dst2 + '/' + filename[0].split('.')[0]+'.xml')
        for filename in val_names:
            src1 = JPG_dir
            dst1 = dir_name + '/val' + '/' + 'JPEGImages'
            src2 = Ann_dir
            dst2 = dir_name + '/val' + '/' + 'Annotations'
            if (not os.path.exists(dst1)):
                os.makedirs(dst1)
            if (not os.path.exists(dst2)):
                os.makedirs(dst2)
            copyfile(src1 + '/' + filename[0], dst1 + '/' + filename[0])
            copyfile(src2 + '/' + filename[0].split('.')[0] + '.xml', dst2 + '/' + filename[0].split('.')[0] + '.xml')

if __name__ == '__main__':
    csv_filename=".csv"
    data_dir="helmet_5_test_data1"
    get_training_and_testing(csv_filename,data_dir)