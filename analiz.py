# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:57:08 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('telefonveriseti.csv')
data.head()

data.info() #Veri setimiz hakkında genel bir bilgi edinelim
# veri setimizdeki eksik değerlerin sayısını öğrenelim ve azalan şekilde sıralayalım.
print(data.isnull().sum().sort_values(ascending=False))

print(data.describe()) #Sutunların istatiksel özelliklerine bakalım

#Asagıda panda  "TotalCharges" sütunundaki tüm değerlerin float64 türü olduğunu
# tespit edemediği için  sütunda sayısal olmayan veriler verileri sayısal bir türe
#donusturduk
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors = 'coerce')
data.loc[data['TotalCharges'].isna()==True]

data[data['TotalCharges'].isna()==True] = 0
data['OnlineBackup'].unique()

# kategorik değerleri sayısal değerlere dönüştürtük 
#algoritmalarımızı uygulayabilmek için

data['gender'].replace(['Male','Female'],[0,1],inplace=True)
data['Partner'].replace(['Yes','No'],[1,0],inplace=True)
data['Dependents'].replace(['Yes','No'],[1,0],inplace=True)
data['PhoneService'].replace(['Yes','No'],[1,0],inplace=True)
data['MultipleLines'].replace(['No phone service','No', 'Yes'],[0,0,1],inplace=True)
data['InternetService'].replace(['No','DSL','Fiber optic'],[0,1,2],inplace=True)
data['OnlineSecurity'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['OnlineBackup'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['DeviceProtection'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['TechSupport'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['StreamingTV'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['StreamingMovies'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['Contract'].replace(['Month-to-month', 'One year', 'Two year'],[0,1,2],inplace=True)
data['PaperlessBilling'].replace(['Yes','No'],[1,0],inplace=True)
data['PaymentMethod'].replace(['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'],[0,1,2,3],inplace=True)
data['Churn'].replace(['Yes','No'],[1,0],inplace=True)

data.pop('customerID') #CutomerId yi listemizden çıkardık
data.info()  #Genel özelliklere baktık

#Tahmini karma modelimizde verinin hangi özelliklerine yer vereceğine
# karar vermek için, karma ve her müşteri özelliği arasındaki ilişkiyi inceleyeceğiz.

corr = data.corr()#veri çerçevesindeki tüm sütunların çift yönlü korelasyonunu bulmak için kullanılır.
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
#Burada verileri 2 boyutlu grafik şeklinde gösterirmek için kullandık
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10) # x eksenini ayarlıyoruz
plt.yticks(fontsize=10)
plt.show()


#Modellerimizdeki dengesiz katsayıların tahmin edilmesini önlemek için, 
#hem “Tenure” hem de “MonthlyCharges”  ile yüksek oranda ilişkili olduğu için
# “TotalCharges” değişkenini düşüreceğiz.
data.pop('TotalCharges')


# Burada Müşteri kaybını öngörmek için birkaç farklı modeli ele alacağız.
 # Verilerimize aşırı derecede uyulmamamız için,
# 7.043 müşteri kayıtlarını bir eğitim ve test setine bölerek 
 # test setinin toplam kayıtların % 25'ini oluşturacağız.
 
 
from sklearn.model_selection import train_test_split#Logisttic algosu burdan calismaya baslar
  #Burada verilerimizi egitim ve test
 #verilerine ayırmak için numpy kutuphanesinin sklearn modelini kullandım
train, test = train_test_split(data, test_size = 0.25)

train_y = train['Churn']
test_y = test['Churn']

train_x = train
train_x.pop('Churn')
test_x = test
test_x.pop('Churn')

#LogisticRegresyon

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
 
logisticRegr = LogisticRegression()
logisticRegr.fit(X=train_x, y=train_y)
 
test_y_pred = logisticRegr.predict(test_x)
confusion_matrix = confusion_matrix(test_y, test_y_pred)
print('Intercept: ' + str(logisticRegr.intercept_))
print('Regression: ' + str(logisticRegr.coef_))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr.score(test_x, test_y)))
print(classification_report(test_y, test_y_pred))
 
confusion_matrix_df = pd.DataFrame(confusion_matrix, ('No churn', 'Churn'), ('No churn', 'Churn'))
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize = 14)
plt.ylabel('Dogru Etiket', fontsize = 14)
plt.xlabel('Tahmini Etiket', fontsize = 14)


#Lojistik regresyon sınıflandırıcımızdan% 81 sınıflandırma doğruluğu elde ettik. 
#Ancak pozitif sınıftaki (kayıp) tahminlerin kesinliği ve hatırlanması göreceli olarak düşüktür, 
#bu da veri setimizin dengesiz olabileceğini göstermektedir.



data['Churn'].value_counts()


#Veri setini dengelemek için, azınlık sınıfından gözlemleri rastgele çoğaltabiliriz.

from sklearn.utils import resample
 
data_majority = data[data['Churn']==0]
data_minority = data[data['Churn']==1]
 
data_minority_upsampled = resample(data_minority,
replace=True,
n_samples=5174, #same number of samples as majority classe
random_state=1) #set the seed for random resampling
# Combine resampled results
data_upsampled = pd.concat([data_majority, data_minority_upsampled])
 
data_upsampled['Churn'].value_counts()


train, test = train_test_split(data_upsampled, test_size = 0.25)
 
train_y_upsampled = train['Churn']
test_y_upsampled = test['Churn']
 
train_x_upsampled = train
train_x_upsampled.pop('Churn')
test_x_upsampled = test
test_x_upsampled.pop('Churn')
 
logisticRegr_balanced = LogisticRegression()
logisticRegr_balanced.fit(X=train_x_upsampled, y=train_y_upsampled)
 
test_y_pred_balanced = logisticRegr_balanced.predict(test_x_upsampled)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr_balanced.score(test_x_upsampled, test_y_upsampled)))
print(classification_report(test_y_upsampled, test_y_pred_balanced))





#Karar Agacları

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix as cm


# Veriyi eğitim ve test alt-veri setlerine ayırma.
# Karar ağacı modeli oluşturma.
# Modeli eğitim verisine ‘fit’ etme.
X_train, X_test, y_train, y_test = train_test_split(train_x, y, test_size=0.3, random_state=42)
d_tree1 = DecisionTreeRegressor(max_depth = 3, random_state=42)
d_tree1.fit(X_train, y_train)


#. Burada Gerçek değerle tahmin arasındaki benzerliğe göre mean absolute error ve accuracy hesapladım

predictions = d_tree1.predict(X_test)
errors = abs(predictions - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'unit.')
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 3), '%.')


# Karar ağacını görselleştirme işlemi
#Sınıflandırma modeli kurulurken yapılan öznitelik önem sıralamasını görselleştirmesi 
from ipywidgets import Image
from io import StringIO
import pydotplus
from sklearn.tree import export_graphviz

dot_data = StringIO()
export_graphviz(d_tree1, feature_names = X.columns,
               out_file = dot_data, filled = True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(value = graph.create_png())
#Burada derinliği 8 olan yeni bir model kurup o modelin başarı oranına bakıyoruz ve öznitelik sıralamasını görselleştiriyoruz.

d_tree2 = DecisionTreeRegressor(max_depth = 8, random_state=42)
d_tree2.fit(X_train, y_train)
predictions = d_tree2.predict(X_test)

errors = abs(predictions - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'unit.')
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 3), '%.')

plt.figure(figsize=(16, 9))

ranking = d_tree2.feature_importances_
features = np.argsort(ranking)[::-1][:10]
columns = X.columns

plt.title("Feature importances based on Decision Tree Regressor", y = 1.03, size = 18)
plt.bar(range(len(features)), ranking[features], color="lime", align="center")
plt.xticks(range(len(features)), columns[features], rotation=80)
plt.show()




#Rastgele Orman algoritması

from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier()
randomForest.fit(train_x, train_y)
print('Accuracy of random forest classifier on test set: {:.2f}'.format(randomForest.score(test_x, test_y)))






