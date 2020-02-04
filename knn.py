# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:43:01 2020

@author: kasum
"""

import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


'''chossing k
1. sqrt of n and if its even you add or subtract 1
2.knn used when data is labelled and noise free
3.small dataset
'''
features=['sepal length (cm)','petal length (cm)','species','petal width (cm)']
data=df[features]
log=data[(data['species']=='setosa') |(data['species']=='virginica')]


#split into train test
y=log['species']
X=log[['sepal length (cm)','petal length (cm)','petal width (cm)']]

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0,test_size=0.2)


#Feature scaling
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


#Building KNN Classifier
import math
k= math.sqrt(len(y))  #picking the number for K. 
#If the value of k is even, you either add or subtract one to get an odd number
#The value of k is assigned to the n_neighbours in the KNN classifier


clf=KNeighborsClassifier(n_neighbors=9, p=2, metric='euclidean') #p=2 just says its a binary classification case
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

#Evaluate me
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)
f1_score(y_test,y_pred)

#make predictions on new data
y_pred=clf.predict(X_test)
