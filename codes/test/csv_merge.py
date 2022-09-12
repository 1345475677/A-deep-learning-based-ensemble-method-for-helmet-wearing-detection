import pandas as pd
import numpy as np


#pd.set_option('display.max_rows', None)
filename1="test_result_method1.csv"
filename2="test_result_method2.csv"
filename3="test_result_modify.csv"
filename4="helmet_test.csv"
path_to_save="test_result_final1.csv"
'''
file1=pd.read_csv(filename1,encoding='gbk')
file2=pd.read_csv(filename2,encoding='gbk')
file3=pd.read_csv(filename3,encoding='gbk')
file4=pd.read_csv(filename4,encoding='gbk')
df1=pd.DataFrame(file1)
df2=pd.DataFrame(file2)
df3=pd.DataFrame(file3)
df4=pd.DataFrame(file4)

df3.insert(len(df3.columns),column='x1',value=1)
df3.insert(len(df3.columns),column='x2',value=np.NAN)
df3.insert(len(df3.columns),column='x3',value=np.NAN)
df3.insert(len(df3.columns),column='y',value=np.NAN)
x=pd.DataFrame(columns=df3.columns)
'''

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

def merge(df1,df2,column_name):
    df2.insert(loc=len(df2.columns),column='exist',value=0)
    val=0.4
    j1=0
    j2=0
    i=0
    while(True):
        if j2 <len(df2):
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
            while(df1.loc[i,'filename']<df2.loc[j1,'filename']):
                j2=j1
                while(df2.loc[j1, 'filename'] == df2.loc[j2, 'filename']):
                    j1-=1
                    if(j1==0):
                        break
        flag = 0
        for j in range(j1, j2):
            dif3 = area(df1.loc[i], df2.loc[j])
            if dif3 > 1 - val and df1.loc[i, 'class'] == df2.loc[j, 'class']:
                flag = 1
                df1.loc[i,column_name]=df2.loc[j,'score']
                df2.loc[j,'exist']=1
                #break
            elif dif3 > 1 - val and df1.loc[i, 'class'] != df2.loc[j, 'class']:
                if flag==1 and df2.loc[j,'exist']==1:
                    break
                else:
                    df2.loc[j, 'exist'] = 1
                    flag = 2
                    df1.loc[i,column_name]=1-df2.loc[j,'score']
        if(i<len(df1)-1):
            if( df1.loc[i,'filename']!=df1.loc[i+1,'filename']):
                x=pd.DataFrame(columns=df1.columns)
                k=0
                for j in range(j1,j2):
                    if(df2.loc[j,'exist']==0):
                        df2.loc[j,'exist']=1
                        x.loc[k,'filename']=df2.loc[j,'filename']
                        x.loc[k,'class']=df2.loc[j,'class']
                        x.loc[k,'xmin']=df2.loc[j,'xmin']
                        x.loc[k,'ymin']=df2.loc[j,'ymin']
                        x.loc[k,'xmax']=df2.loc[j,'xmax']
                        x.loc[k,'ymax']=df2.loc[j,'ymax']
                        x.loc[k,column_name]=df2.loc[j,'score']
                        k+=1
                if(len(x)>0):
                    #print(x)
                    above=df1.loc[:i]
                    below=df1.loc[i+1:]
                    df1=above.append(x,ignore_index=True).append(below,ignore_index=True)
                    while(df1.loc[i,'filename']==df1.loc[i+1,'filename']):
                        i+=1
        elif(i==len(df1)-1):
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
                    x.loc[k,column_name]=df2.loc[j,'score']
                    k += 1
            if (len(x) > 0):
                above = df1.loc[:i]
                below = df1.loc[i + 1:]
                df1 = above.append(x, ignore_index=True).append(below, ignore_index=True)
                i=len(df1)-1
        if i<len(df1):
            i+=1
        if i==len(df1):
            break
    return df1

def merge2(df1,df2):
    val = 0.7
    j1 = 0
    j2 = 0
    for i in range(len(df1)):
        if j2 < len(df2):
            while (df2.loc[j1, 'filename'] == df2.loc[j2, 'filename']):
                j2 += 1
                if j2 == len(df2):
                    break
            if (df1.loc[i, 'filename'] > df2.loc[j1, 'filename']):
                j1 = j2
                while (df2.loc[j1, 'filename'] == df2.loc[j2, 'filename']):
                    j2 += 1
                    if j2 == len(df2):
                        break
        flag = 0
        for j in range(j1, j2):
            dif3 = area(df1.loc[i], df2.loc[j])
            if dif3 > 1 - val and df1.loc[i, 'class'] == df2.loc[j, 'class']:
                flag = 1
                break
        if (flag == 1):
            df1.loc[i,'y']=1
        else:
            df1.loc[i,'y']=0
    return df1

def merge_for_video(df1,df2,column_name='score2'):
    df2.insert(loc=len(df2.columns),column='exist',value=0)
    val=0.4
    j1=0
    j2=0
    i=0
    while(True):
        if j2 <len(df2):
            while (df2.loc[j1, 'frame'] == df2.loc[j2, 'frame']):
                j2 += 1
                if j2 == len(df2):
                    break
            while (df1.loc[i, 'frame'] > df2.loc[j1, 'frame']):
                j1 = j2
                while (df2.loc[j1, 'frame'] == df2.loc[j2, 'frame']):
                    j2 += 1
                    if j2 == len(df2):
                        break
            while(df1.loc[i,'frame']<df2.loc[j1,'frame']):
                j2=j1
                while(df2.loc[j1, 'frame'] == df2.loc[j2, 'frame']):
                    j1-=1
                    if(j1==0):
                        break
        flag = 0
        for j in range(j1, j2):
            dif3 = area(df1.loc[i], df2.loc[j])
            if dif3 > 1 - val and df1.loc[i, 'class'] == df2.loc[j, 'class']:
                flag = 1
                df1.loc[i,column_name]=df2.loc[j,'score']
                df2.loc[j,'exist']=1
                #break
            elif dif3 > 1 - val and df1.loc[i, 'class'] != df2.loc[j, 'class']:
                if flag==1 and df2.loc[j,'exist']==1:
                    break
                else:
                    df2.loc[j, 'exist'] = 1
                    flag = 2
                    df1.loc[i,column_name]=1-df2.loc[j,'score']
        if(i<len(df1)-1):
            if( df1.loc[i,'frame']!=df1.loc[i+1,'frame']):
                x=pd.DataFrame(columns=df1.columns)
                k=0
                for j in range(j1,j2):
                    if(df2.loc[j,'exist']==0):
                        df2.loc[j,'exist']=1
                        x.loc[k,'frame']=df2.loc[j,'frame']
                        x.loc[k,'class']=df2.loc[j,'class']
                        x.loc[k,'xmin']=df2.loc[j,'xmin']
                        x.loc[k,'ymin']=df2.loc[j,'ymin']
                        x.loc[k,'xmax']=df2.loc[j,'xmax']
                        x.loc[k,'ymax']=df2.loc[j,'ymax']
                        x.loc[k,column_name]=df2.loc[j,'score']
                        k+=1
                if(len(x)>0):
                    #print(x)
                    above=df1.loc[:i]
                    below=df1.loc[i+1:]
                    df1=above.append(x,ignore_index=True).append(below,ignore_index=True)
                    while(df1.loc[i,'frame']==df1.loc[i+1,'frame']):
                        i+=1
        elif(i==len(df1)-1):
            x = pd.DataFrame(columns=df1.columns)
            k = 0
            for j in range(j1, j2):
                if (df2.loc[j, 'exist'] == 0):
                    df2.loc[j, 'exist'] = 1
                    x.loc[k, 'frame'] = df2.loc[j, 'frame']
                    x.loc[k, 'class'] = df2.loc[j, 'class']
                    x.loc[k, 'xmin'] = df2.loc[j, 'xmin']
                    x.loc[k, 'ymin'] = df2.loc[j, 'ymin']
                    x.loc[k, 'xmax'] = df2.loc[j, 'xmax']
                    x.loc[k, 'ymax'] = df2.loc[j, 'ymax']
                    x.loc[k,column_name]=df2.loc[j,'score']
                    k += 1
            if (len(x) > 0):
                above = df1.loc[:i]
                below = df1.loc[i + 1:]
                df1 = above.append(x, ignore_index=True).append(below, ignore_index=True)
                i=len(df1)-1
                #while(df1.loc[i,'frame']==df1.loc[i+1,'frame']):
                 #   i+=1
        if i<len(df1):
            i+=1
        if i==len(df1):
            break
    return df1




def my_merge(df1,df2,df3,df4):
    print(len(df3))
    df3 = merge(df3, df1, column_name='x2')
    print(len(df3))
    df3 = merge(df3, df2, column_name='x3')
    print(len(df3))
    df3 = merge2(df3, df4)
    df3.to_csv(path_to_save, index=False)

if __name__=="__main__":
    file1=pd.read_csv(filename3,encoding='gbk')
    df1=pd.DataFrame(file1)
    file2=pd.read_csv("test_result_face_expand_judged.csv",encoding='gbk')
    df2=pd.DataFrame(file2)
    file4=pd.read_csv(filename4,encoding='gbk')
    df4=pd.DataFrame(file4)
    df=merge(df1,df2,'score2')
    print(df1)
    print(df2)
    print(df)
    df=merge2(df,df4)
    print(df)
    df.insert(loc=6, column='area', value=0)
    for i in range(len(df)):
        df.loc[i,'area']=(df.loc[i,'xmax']-df.loc[i,'xmin'])*(df.loc[i,'ymax']-df.loc[i,'ymin'])
    print(df)
    df.to_csv(path_to_save,index=False)




#my_merge(df1,df2,df3,df4)

