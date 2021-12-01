# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 21:07:23 2019

@author: Administrator
"""
import numpy as np #Pandas serilerini Numpy dizileri ile karşılaştırabilmek için Numpy kütüphanesini de import ettik.
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from PIL import  Image
import seaborn as sns#visualization
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn import preprocessing





df_xlsx=pd.read_excel('e-tablo.xlsx')


df=pd.read_csv('telefonveriseti.csv')

print ("Rows     : " ,df.shape[0])
print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())


print ("\nMissing values :  ", df.isnull().sum().values.sum())


print ("\nUnique values :  \n",df.nunique())









print("eksik listeleri kaldırır")
df.dropna(inplace = True)
print(df.shape)
print(df.dtypes)


print("df hakıında genel bilgi verir")
df.info() 
#info() metodu ile ;0' dan 7043 a kadar numaralandırılmış 7043 girdi olduğu,
# her bir sütunda null olmayan girdi sayısı ve sütunun veri tipi gibi
# bilgileri elde edebiliriz.


print(df) # tum satirlari gosterir
print(df.head(3))  #Bastan ilk 3 satiri gosterir
print(df.tail(3))   #sondan 3 satiri gosteri
print("sadece sutunları gosterir")
print(df.columns) 
print(df['customerID'][0:5])# ilk 5 kullanıcııd listelenir

print (df.customerID)  #Tum customerId leri listeler

print(df[['customerID','gender','Partner']]) #bu sekilde istedigimiz kolonları listeledik
print(df.head(4))   # 4 satir listeledik
#veri decme islemini d.floc[row,column] ile yapariz
print(df.iloc[1:4])  #☺ ilk indisi getirmedi
print(df.iloc[2:1])

for index,row in df.iterrows():
    print(index,row['customerID'])
    
    
print("cinsiyeti erkek olanları listeler")
df.loc[df['gender']=="Male"]  

print(df.isnull().sum().sort_values(ascending=False))#eksik bilgileri gosterir

print("verilerin tür değişkenine göre dağılımı")
print(df.groupby('Churn').size())




print('Verinin özeti:')
print(df.describe()) #Sutunların istatiksel ozelliklerine bakılır
#describe() metodu sayısal verilere sahip olan sütunların
 #max, min , std…gibi istatiksel değerlerini döndürür. 
#Bizim veri setimizde sayısal olan sütünlar “SeniorCitizen”
# "tenure" "MonthlyCharges" sütünlerinin istatiksel özetini görebiliyoruz.


print("her bir degerin sutunda bulunma sayısı")
df["Churn"].value_counts() 

print(df["gender"].describe())
#İstediğimiz takdirde describe() metodunu string verilerde de kullanabiliriz.
#count:NULL olmayan girdi sayısi
#unique:birbirinden farklı kac kategori oldugu bilgisi
#top:en cok bulunan kategori ad
#freq:en cok bulunan kategorinin sutunda bulunma sıklıgı



df.describe(include=['O'])


perc =[.20, .40, .60, .80] #yüzdelik dilimler
include =['object', 'float', 'int'] #dahil edilecek degerler
desc = df.describe(percentiles = perc, include = include) 
desc #görüntüle



#Verinin gorsellestirilmesi
print("Kutu grafigi")
df.plot(kind='box', subplots=True, sharex=False, sharey=False)
plt.show()

print("histogram")
df.hist()
plt.show()














































