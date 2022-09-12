import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
#filename1="test_result_method1.csv"
filename2="test_modify2.csv"
filename3="test_modify.csv"
filename4="helmet_test.csv"
path_to_save="test_final1.csv"
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

def merge(df1,df2,column_name):#insert df1 to df2
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
        if j2 <len(df2):#保证df1和df2的filename相同，确定j的区间[j1,j2]
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




def my_merge():
    def cmp(a1, b1):
        minlen = min(len(a1), len(b1))
        a = []
        b = []
        for x in a1:
            a.append(x)
        for x in b1:
            b.append(x)
        if (a[0] == 'P'):
            a[0] = 'p'
        if (b[0] == 'P'):
            b[0] = 'p'
        for i in range(minlen):
            if (a[i] > b[i]):
                return 1
            elif (a[i] < b[i]):
                return -1
        return 0
    def merge2(df1,df2):
        df2.insert(loc=len(df2.columns), column='exist', value=0)
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
                    df1.loc[i, 'score'] = df2.loc[j, 'score']
                    df1.loc[i, 'score2'] = df2.loc[j, 'score2']
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
    def merge1(df1, df2, column_name):  #insert df2 to df1
        df2.insert(loc=len(df2.columns), column='exist', value=0)
        val = 0.4
        j1 = 0
        j2 = 0
        i = 0
        while (True):
            if j2 < len(df2):
                while (cmp(df1.loc[i, 'filename'],df2.loc[j1, 'filename'])==1):
                    j1+=1
                if(cmp(df1.loc[i, 'filename'],df2.loc[j1, 'filename'])==-1):
                    while(cmp(df1.loc[i, 'filename'],df2.loc[j1, 'filename'])==-1):
                        i+=1
                while (cmp(df1.loc[i, 'filename'],df2.loc[j1, 'filename'])==-1):
                    j1-=1
                j2=j1
                if df1.loc[i, 'filename'] != df2.loc[j1, 'filename']:
                    print("!!!", df1.loc[i, 'filename'], df2.loc[j1, 'filename'])
                while (df2.loc[j1, 'filename'] == df2.loc[j2, 'filename']):
                    j2 += 1
                    if j2 == len(df2):
                        break


            flag = 0
            for j in range(j1, j2):
                if(df1.loc[i, 'filename']!=df2.loc[j,'filename']):
                    print(df1.loc[i, 'filename'],df2.loc[j,'filename'])
                if flag==1:
                    break
                dif3 = area(df1.loc[i], df2.loc[j])
                if dif3 > 1 - val and df1.loc[i, 'class'] == df2.loc[j, 'class']:
                    if df2.loc[j,'exist']==0:
                        flag = 1
                        df1.loc[i, column_name] = df2.loc[j, 'score']
                        df2.loc[j, 'exist'] = 1
                        # break
                    else:
                        continue
                elif dif3 > 1 - val and df1.loc[i, 'class'] != df2.loc[j, 'class']:
                    if  df2.loc[j,'exist']==0:
                        df2.loc[j, 'exist'] = 1
                        flag = 2
                        df1.loc[i, column_name] = 1 - df2.loc[j, 'score']

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
                            x.loc[k, column_name] = df2.loc[j, 'score']
                            k += 1
                    if (len(x) > 0):
                        # print(x)
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
                        x.loc[k, column_name] = df2.loc[j, 'score']
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
    filename1='.csv'
    filename2='.csv'
    filename3='.csv'
    file1=pd.read_csv(filename1,encoding='gbk')
    file2=pd.read_csv(filename2,encoding='gbk')
    file3=pd.read_csv(filename3,encoding='gbk')
    df1=pd.DataFrame(file1)
    df2=pd.DataFrame(file2)
    df3=pd.DataFrame(file3)
    df3.insert(loc=len(df3.columns), column='tg', value=1)
    df=merge1(df3,df1,column_name='score')
    df=merge1(df,df2,column_name='score2')
    df.insert(loc=6, column='area', value=0)
    for i in range(len(df)):
        df.loc[i,'area']=(df.loc[i,'xmax']-df.loc[i,'xmin'])*(df.loc[i,'ymax']-df.loc[i,'ymin'])
    df.to_csv('.csv')
# if __name__=="__main__":
#     file1=pd.read_csv(filename3,encoding='gbk')
#     df1=pd.DataFrame(file1)
#     file2=pd.read_csv(filename2,encoding='gbk')
#     df2=pd.DataFrame(file2)
#     # file4=pd.read_csv(filename4,encoding='gbk')
#     # df4=pd.DataFrame(file4)
#     df=merge(df1,df2,'score2')
#     # df=merge2(df,df4)
#     df.insert(loc=6, column='area', value=0)
#     for i in range(len(df)):
#         df.loc[i,'area']=(df.loc[i,'xmax']-df.loc[i,'xmin'])*(df.loc[i,'ymax']-df.loc[i,'ymin'])
#     print(df)
#     df.to_csv(path_to_save,index=False)

def my_merge2():
    def cmp(a1, b1):
        minlen = min(len(a1), len(b1))
        a = []
        b = []
        for x in a1:
            a.append(x)
        for x in b1:
            b.append(x)
        if (a[0] == 'P'):
            a[0] = 'p'
        if (b[0] == 'P'):
            b[0] = 'p'
        for i in range(minlen):
            if (a[i] > b[i]):
                return 1
            elif (a[i] < b[i]):
                return -1
        return 0

    def merge2(df1, df2):
        df2.insert(loc=len(df2.columns), column='exist', value=0)
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
                    df1.loc[i, 'score'] = df2.loc[j, 'score']
                    df1.loc[i, 'score2'] = df2.loc[j, 'score2']
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

    def merge1(df1, df2, column_name):
        df2.insert(loc=len(df2.columns), column='exist', value=0)
        val = 0.4
        j1 = 0
        j2 = 0
        i = 0
        while (True):
            if j2 < len(df2):
                while (cmp(df1.loc[i, 'filename'], df2.loc[j1, 'filename']) == 1):
                    j1 += 1
                if (cmp(df1.loc[i, 'filename'], df2.loc[j1, 'filename']) == -1):
                    while (cmp(df1.loc[i, 'filename'], df2.loc[j1, 'filename']) == -1):
                        i += 1
                while (cmp(df1.loc[i, 'filename'], df2.loc[j1, 'filename']) == -1):
                    j1 -= 1
                j2 = j1
                if df1.loc[i, 'filename'] != df2.loc[j1, 'filename']:
                    print("!!!", df1.loc[i, 'filename'], df2.loc[j1, 'filename'])
                while (df2.loc[j1, 'filename'] == df2.loc[j2, 'filename']):
                    j2 += 1
                    if j2 == len(df2):
                        break

            flag = 0
            for j in range(j1, j2):
                if (df1.loc[i, 'filename'] != df2.loc[j, 'filename']):
                    print(df1.loc[i, 'filename'], df2.loc[j, 'filename'])
                if flag == 1:
                    break
                dif3 = area(df1.loc[i], df2.loc[j])
                if dif3 > 1 - val and df1.loc[i, 'class'] == df2.loc[j, 'class']:
                    if df2.loc[j, 'exist'] == 0:
                        flag = 1
                        df1.loc[i, column_name] = df2.loc[j, 'score']
                        df2.loc[j, 'exist'] = 1
                        # break
                    else:
                        continue
                elif dif3 > 1 - val and df1.loc[i, 'class'] != df2.loc[j, 'class']:
                    if df2.loc[j, 'exist'] == 0:
                        df2.loc[j, 'exist'] = 1
                        flag = 2
                        df1.loc[i, column_name] = 1 - df2.loc[j, 'score']

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
                            x.loc[k, column_name] = df2.loc[j, 'score']
                            k += 1
                    if (len(x) > 0):
                        # print(x)
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
                        x.loc[k, column_name] = df2.loc[j, 'score']
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

    filename1 = '.csv'
    filename2 = '.csv'
    file1 = pd.read_csv(filename1, encoding='gbk')
    file2 = pd.read_csv(filename2, encoding='gbk')
    df1 = pd.DataFrame(file1)
    df2 = pd.DataFrame(file2)
    df2.insert(loc=len(df2.columns), column='tg', value=1)
    df = merge1(df2, df1, column_name='score')
    df.insert(loc=6, column='area', value=0)
    for i in range(len(df)):
        df.loc[i, 'area'] = (df.loc[i, 'xmax'] - df.loc[i, 'xmin']) * (df.loc[i, 'ymax'] - df.loc[i, 'ymin'])
    df.to_csv('.csv')

if __name__ == '__main__':
    my_merge()
    # my_merge2()