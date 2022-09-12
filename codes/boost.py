import pandas as pd
import numpy as np
import random

pd.set_option('display.max_rows', None)

def myrandom():
    return random.randint(1,10)

filename=".csv"
path_to_save=".csv"
def set_data(filename):
    file=pd.read_csv(filename,encoding='gbk')
    df=pd.DataFrame(file)
    df.insert(loc=len(df.columns),column='label',value=0)
    for i in range(len(df)):
        if(df.loc[i,'x1']==1 and df.loc[i,'x2']==1 or df.loc[i,'x3']==1):
            df.loc[i,'label']=1
        elif(df.loc[i,'x1']==1 and df.loc[i,'x2']==2 and df.loc[i,'x3']==1):
            df.loc[i,'label']=1
        elif(df.loc[i,'x1']==1 and df.loc[i,'x2']!=1 and df.loc[i,'x3']!=1):
            df.loc[i,'label']=1
        elif(df.loc[i,'x1']!=1 and df.loc[i,'x2']==1 ):
            j1=i
            while(df.loc[j1-1,'filename']==df.loc[i,'filename']):
                j1-=1
            sum=0
            j2=j1
            for j2 in range(j1,i):
                if(df.loc[j2,'x1']!=1):
                    break
            for j in range(j1,j2):
                area=(df.loc[j,'xmax']-df.loc[j,'xmin'])*(df.loc[j,'ymax']-df.loc[j,'ymin'])
                sum+=area
            if(j2>j1):
                area=sum/(j2-j1)
            else:
                area=900
            if((df.loc[i,'xmax']-df.loc[i,'xmin'])*(df.loc[i,'ymax']-df.loc[i,'ymin'])<0.8*area):
                df.loc[i,'label']=1
            else:
                df.loc[i,'label']=0
    return df

mydf=set_data(filename)
for i in range(len(mydf)):
    if(mydf.loc[i,'label']==0):
        mydf=mydf.drop(i)
print(mydf)
mydf.to_csv(path_to_save,index=False)

