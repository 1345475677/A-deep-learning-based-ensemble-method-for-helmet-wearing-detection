import numpy as np
import pandas as pd




def mysort(df):
    val=0.001
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
        df.loc[i]=df.loc[minloc]
        df.loc[minloc]=temp
    return df

def myvalue(df1,df2):
    if len(df1)>len(df2):
        mylen=len(df1)
    else:
        mylen=len(df2)

    val=0.3
    real=0
    more=0
    less=0
    erro=0
    i=0
    j=0
    while(True):
        if i==len(df1)-1 or j==len(df2)-1:
            break
        #else:
         #   print("i=",i,"j=",j)
          #  print(df1.loc[i,'filename'],df2.loc[j,'filename'])
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


#more,less,erro,real=myvalue(df1,df2)
#print("more=",more,"less=",less,"erro=",erro,"real=",real)


#df1=mysort(df1)
#df2=mysort(df2)
#more,less,erro,real=myvalue(df1,df2)
#print("more=",more,"less=",less,"erro=",erro,"real=",real)




            
file1=pd.read_csv("D:/service2019/aaa.csv",encoding='gbk')

df1=pd.DataFrame(file1)
df1=mysort(df1)
df1.to_csv("D:/service2019/bbb.csv")