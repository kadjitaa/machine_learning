# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 12:38:14 2020

@author: kasum
"""

#Standardization - data must first be standardised before performing PCA
from sklearn.preprocessing import StandardScaler
train_x=StandardScaler().fit_transform(df[features]) #Fit and transform training set only
test_x=StandardScaler().transform() #transform test set

from sklearn.decomposition import PCA
pca=PCA(0.95) # this essesntially means you what the algorithm to choose the number of components for which 95% of variance in the data is captured
#pca.fit_transform(train_x) # the input to this must be values that have already been standardised train

'''the fit and tranform step can be done seperately or merged as shown above'''

pca.fit(train_x) # only the training data must be used for fitting in both instances
pca.explained_variance_ratio_ #gives  you the fit accuraccy in terms of explained variance but not needed if the input to PCA already states the desired accuracy

#transform both datasets
train_x=pca.transform(train_x)
test_x=pca.transform(train_y)
