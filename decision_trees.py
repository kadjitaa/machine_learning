# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:52:33 2020

@author: kasum
"""

##############################################################################
### DECISION TREES_simplilearn
##############################################################################
import numpy as np
import pandas as pd
from pylab import *

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#loading file
data=pd.read_csv(r'C:\Users\kasum\Desktop\COMP551_AppliedML\practice\part_1_data.csv', header=2)

#Seperating the target variable

x=data.values[:,1:]
y=data.values[:,0]

#Splitting into test and train
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.35, random_state=100)

#Build decision tree
clf_entropy=DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=6, min_samples_leaf=6)
clf_entropy.fit(x_train, y_train)

#Making Predictions on test data
y_pred=clf_entropy.predict(x_test)

#Checking Prediction Accuracy
pred_accuracy=accuracy_score(y_test, y_pred)*100; print('The accuracy is '+ str(pred_accuracy)) # by multiplying it with 100 you get the accuracy in %


#Making Predictions on new sample
y_pred=clf_entropy.predict(data.values[0:49,1:]); print(y_pred)
