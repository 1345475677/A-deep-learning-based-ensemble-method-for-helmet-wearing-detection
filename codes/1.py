
import numpy as np
import pandas as pd

#pd.set_option('display.max_rows', None)
#文件路径
filename1=".csv"
filename2=".csv"
filename3=".csv"
path_to_save=".csv"
csv_final=".csv"
csv_boost=".csv"
csv_boost_cf=".csv"

def mysort(df):
    val=0.1
    for i in range(len(df)):
        myfilename=df.loc[i,'filename']
        minloc=i
        myxmid=df.loc[i,'xmid']
        myymid=df.loc[i,'ymid']
        if df.loc[i,'class'] == np.NAN:
            continue
        for j in range(i,len(df)):
            if(df.loc[j,'filename'] == myfilename):
                dif1=abs((df.loc[j,'xmid']-myxmid)/(df.loc[i,'xmax']-df.loc[i,'xmin']))
                dif2=abs((df.loc[j,'xmid']-myxmid)/(df.loc[j,'xmax']-df.loc[j,'xmin']))
                if(dif1>val or dif2>val):
                    if myxmid>df.loc[j,'xmid']:
                        minloc=j
                        myxmid=df.loc[j,'xmid']
                else:
                    if myymid>df.loc[j,'ymid']:
                        minloc=j
                        myymid=df.loc[j,'ymid']
            else:
                break
        temp=np.array(df.loc[i])
        df.loc[i]=np.array(df.loc[minloc])
        df.loc[minloc]=temp
    return df

def myvalue(df1,df2):
    if len(df1)>len(df2):
        mylen=len(df1)
    else:
        mylen=len(df2)

    val=0.3#评判标准
    real=0
    more=0
    less=0
    erro=0
    i=0
    j=0
    while(True):
        if i==len(df1)-1 or j==len(df2)-1:
            break
        if df1.loc[i,'filename'] == df2.loc[j,'filename']:
            dif1=abs((df1.loc[i,'xmid']-df2.loc[j,'xmid'])/(df1.loc[i,'xmax']-df1.loc[i,'xmin']))
            dif2=abs((df1.loc[i,'ymid']-df2.loc[j,'ymid'])/(df1.loc[i,'ymax']-df1.loc[i,'ymin']))
            if dif1>val or dif2>val:
                if dif1>val:
                    if df1.loc[i,'xmid']>df2.loc[j,'xmid']:
                        j+=1
                        more+=1
                    else:
                        i+=1
                        less+=1
                elif dif2>val:
                    if df1.loc[i,'ymid']>df2.loc[j,'ymid']:
                        j+=1
                        more+=1
                    else:
                        i+=1
                        less+=1
            elif df1.loc[i,'class'] != df2.loc[j,'class']:
                erro+=1
                i+=1
                j+=1
            else :
                real+=1
                i+=1
                j+=1
        else:
            if df1.loc[i,'filename']>df2.loc[j,'filename']:
                for j1 in range(j,mylen):
                    if df2.loc[j,'filename'] != df2.loc[j1,'filename']:
                        more+=j1-j
                        j=j1
                        break

            elif df1.loc[i,'filename']<df2.loc[j,'filename']:
                for i1 in range(i,mylen):
                    if df1.loc[i1,'filename'] != df1.loc[i,'filename']:
                        less += i1 - i
                        i = i1
                        break

    return more,less,erro,real
def myvalue2(df1,df2):
    val = 0.7
    real = 0
    less = 0
    j1=0
    j2=0
    for i in range(len(df1)):
        if j2 <len(df2):
            while (df2.loc[j1, 'filename'] == df2.loc[j2, 'filename']):
                j2 += 1
                if j2==len(df2):
                    break
            while(df1.loc[i,'filename'] > df2.loc[j1,'filename']):
                j1=j2
                while(df2.loc[j1,'filename']==df2.loc[j2,'filename']):
                    j2+=1
                    if j2 == len(df2):
                        break
            while (df1.loc[i, 'filename'] < df2.loc[j1, 'filename']):
                j2 = j1
                while (df2.loc[j1, 'filename'] == df2.loc[j2, 'filename']):
                    j1 -= 1
                    if (j1 == 0):
                        break

        flag=0
        for j in range(j1,j2):
           # dif1 = abs((df1.loc[i, 'xmid'] - df2.loc[j, 'xmid']) / (df1.loc[i, 'xmax'] - df1.loc[i, 'xmin']))
           # dif2 = abs((df1.loc[i, 'ymid'] - df2.loc[j, 'ymid']) / (df1.loc[i, 'ymax'] - df1.loc[i, 'ymin']))
            dif3=area(df1.loc[i],df2.loc[j])

            #if dif1<val and dif2<val and df1.loc[i,'class']==df2.loc[j,'class']:
               # flag=1
              #  break
            if dif3>val and df1.loc[i,'class']==df2.loc[j,'class']:
                flag=1
                break
        if(flag==1):
            real+=1
        else:
            less+=1
    return real,less
#cal Overlapping area
def area(list1,list2):
    xmax1=list1.loc['xmax']
    xmin1=list1.loc['xmin']
    ymin1=list1.loc['ymin']
    ymax1=list1.loc['ymax']
    xmax2=list2.loc['xmax']
    xmin2=list2.loc['xmin']
    ymin2=list2.loc['ymin']
    ymax2=list2.loc['ymax']
    if(xmax1==np.NAN):
        return 0
    xmax3=min(float(xmax1),float(xmax2))
    xmin3=max(float(xmin1),float(xmin2))
    ymax3=min(float(ymax1),float(ymax2))
    ymin3=max(float(ymin1),float(ymin2))

    area1=abs((xmax1-xmin1)*(ymax1-ymin1))
    area2=abs((xmax2-xmin1)*(ymax2-ymin2))
    area3=(xmax3-xmin3)*(ymax3-ymin3)
    min_area=min(area1,area2)
    if(xmax3-xmin3<=0 or ymax3-ymin3<=0):
        return 0
    return area3/min_area


def val(filename1,filename2):
    file1=pd.read_csv(filename1,encoding='gbk')
    file2=pd.read_csv(filename2,encoding='gbk')
    df1=pd.DataFrame(file1)
    df2=pd.DataFrame(file2)
    myvalue2(df1,df2)
    real, less = myvalue2(df1, df2)
    real1, more = myvalue2(df2, df1)
    print( real / len(df2),  more / len(df2),  less / len(df1))

def modify1(filename2,path_to_save):
    file2 = pd.read_csv(filename2, encoding='gbk')
    df2 = pd.DataFrame(file2)
    list2 = np.array(df2)
    str = df2.columns.values
    for i in range(2, 6):
        str[i] = int(str[i])
    list2 = np.insert(arr=list2, values=str, obj=0, axis=0)
    columns_name_2 = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax','score']
    df2 = pd.DataFrame(list2, columns=columns_name_2)
    #xmid2 = (df2.loc[:, 'xmin'] + df2.loc[:, 'xmax']) / 2
    #ymid2 = (df2.loc[:, 'ymin'] + df2.loc[:, 'ymax']) / 2
   # df2['xmid'] = pd.Series(xmid2)
   # df2['ymid'] = pd.Series(ymid2)
    df2.to_csv(path_to_save,index=False)

def modify2(filename,path_to_save):
    val=0.7
    file=pd.read_csv(filename)
    df=pd.DataFrame(file)
    df.insert(1,column="class",value="hat")
    for i in range(len(df)):
        if(df.loc[i,'degree']<val):
            df.drop(index=i)
    df.to_csv(path_to_save)

#modify1(filename2,path_to_save)



#modify1(filename2,filename2)
val(filename1,filename2)

