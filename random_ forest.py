# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:56:14 2020

@author: kasum
"""
"""can estimate missing data
very little over-fitting
multiple decision trees. output of the maj of the decision trees decide on prediction.
very suitable for large datasets
info is the reduction in the entropy following splitting of the data
"""
import pandas as pd
import numpy as np
from functions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
np.random.seed(0) #random seed



###############################################################################
## RANDOM FOREST
###############################################################################

#Loading Data
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['species']=pd.Categorical.from_codes(iris.target, iris.target_names)#not a very necessary step, unless you care about having the name tags

#Test and Train Data
df['is_train']
import random
msk=np.random.rand(len(df))<0.75 
train=df[msk]
test=df[~msk]

#Features 
features=df.columns[:4]

y=pd.factorize(train['species'])[0] #factorze just converts them to 0's 1's 2's based on uniqueness of the items 

#Creating the random forest classifier
clf= RandomForestClassifier(n_jobs=2, random_state=0) #n jobs just sets priority on using the system

#Trainig the classifier
clf.fit(train[features],y)

#Testing the classifier
pred=clf.predict(test[features])

#Viewing predicted probs for the first n features
clf.predict_proba(test[features])[0:10]

#mapping names to predictions
preds=iris.target_names[pred]

#Creating a confusion matrix to assess accuracy
accuracy_mat=pd.crosstab(test['species'], preds, rownames=['Actual Species'],colnames=['Predicted Species'])
perc_accuracy=randforestAccuracy(accuracy_mat)



#Making prediction with new data
pred=clf.predict('place new data here') #remember it must be ab array of array ndarray









