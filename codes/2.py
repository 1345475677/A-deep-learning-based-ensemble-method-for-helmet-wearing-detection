import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

filename='.csv'
file=pd.read_csv(filename,encoding='gbk')
df=pd.DataFrame(file)

darknetlist=[]
tinyfacelist=[]
df=df.fillna(value=0)
for i in range(len(df)):
    # df.loc[i,'area']=df.loc[i,'xmax']-df.loc[i,'xmin']
    if(df.loc[i,'tg']==1):
        if(df.loc[i,'score']>=0.5):
            darknetlist.append(df.loc[i,'area'])
        if(df.loc[i,'score2']>=0.5):
            tinyfacelist.append(df.loc[i,'area'])
darknetlist=np.array(darknetlist)
tinyfacelist=np.array(tinyfacelist)
hist1,bins1 = np.histogram(darknetlist,bins =  [i*50 for i in range(41)])
hist2,bins2 = np.histogram(tinyfacelist,bins =  [i*50 for i in range(41)])
print(hist1)
print(hist2)
plt.hist(darknetlist, bins = [i*50 for i in range(41)])
plt.savefig('.png')
plt.show()
plt.hist(tinyfacelist, bins = [i*50 for i in range(41)])
plt.savefig('.png')
plt.show()