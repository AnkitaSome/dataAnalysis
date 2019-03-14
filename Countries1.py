# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:06:47 2018

@author: Aroj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:57:33 2018

@author: Aroj
"""
import time 
import warnings
import seaborn as sns
import plotly
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

countries = pd.read_csv('countriesworld.csv')
x = countries.iloc[:,[3,5]].values
y = countries.iloc[:,6].values
#splitting into the training and testing set
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)

#feature selection
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#finding the value of parameter k
#The value of k has to be odd to avoid confusion between classes of data
import math
kval = math.sqrt(len(y_test))
if(int(kval) % 2 == 0):
    k = int(kval) - 1
else:
    k = int(kval)
print(k)

#using KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=k,p=2,metric='euclidean')
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

# evaluate accuracy
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred,average=None))


wh1=countries[['Infant mortality','Deathrate']]
cor=wh1.corr()

fig=sns.heatmap(cor,square=True,annot=True)
fig.savefig("a.png")
data=dict(type='choropleth',locations=countries['Country'],locationmode='country names',z=countries['Infant mortality'],text=countries['Country'],colorbar={'title':'Infant Mortality'})
layout=dict(title='Infant Mortality Rate',geo=dict(showframe=False,projection={'type':'orthographic'}))
choromap3=go.Figure(data=[data],layout=layout)
plotly.offline.plot(choromap3,output_type='file',filename='a.html')
data=dict(type='choropleth',locations=countries['Country'],locationmode='country names',z=countries['Deathrate'],text=countries['Country'],colorbar={'title':'Death Rate'})
layout=dict(title='Death Rate',geo=dict(showframe=False,projection={'type':'orthographic'}))
choromap3=go.Figure(data=[data],layout=layout)
plotly.offline.plot(choromap3,output_type='file',filename='b.html')

print("Prediction part:")
#pd = input('Enter population density(per Sq Mile): ')
#gdp = input('Enter GDP ($Per Capita): ')
imr = input('Enter Infant Mortality Rate(per 1000 births): ')
#br = input('Enter Birth Rate: ')
dr = input('Enter Death Rate: ')
dataClass = classifier.predict([[imr,dr]])
print('Prediction: '),
 
if dataClass == 0:
    print('Northern Hemisphere')
else:
    print('Southern Hemisphere')
    
print("mean squared error : %.2f" % np.mean((classifier.predict(x_test)-y_test)**2))    