# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:14:23 2020

@author: kasum
"""
##############################################################################
### DECISION TREES
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



################################################################################
### SIMPLE LINEAR REGRESSION MODELS
###############################################################################
import numpy as np
import pandas as pd
from pylab import *
import sklearn
from sklearn import linear_model
from sklearn.metrics import r2_score

data=pd.read_csv(r'C:\Users\kasum\Desktop\COMP551_AppliedML\practice\FuelConsumption.csv')
#viz = data[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
cdf = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#splitting into test and train
msk = np.random.rand(len(data)) < 0.8 #randomly picks 80% of the data
train = cdf[msk]
test = cdf[~msk]

#build linear model
regr = linear_model.LinearRegression()
train_x = array(train[['ENGINESIZE']])
train_y = array(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)

#The coefficients of the model
coef= regr.coef_
interc= regr.intercept_

#Plot
scatter(train.ENGINESIZE, train.CO2EMISSIONS)
plot(train_x, coef*train_x + interc, '-r')  #the regression best fit line

#####################
#Evaluation of model
####################
#Prediction
test_y_hat = regr.predict(test_x) #when inputing a single value ensure it has the right dimensions

'''linear model can be evaluated with ME (absolute), MSE, RMSE or R-squared'''
test_x = array(test[['ENGINESIZE']])
test_y = array(test[['CO2EMISSIONS']])

me='%.2f' % np.mean(np.absolute(test_y_hat - test_y)) #Mean abs err
mse='%.2f' % np.mean((test_y_hat - test_y) ** 2) #Mean Squared Err
r2='%.2f' % r2_score(test_y_hat , test_y) #best possible score is 1  and when negative it means the model is terribly worse



################################################################################
### MULTIPLE LINEAR REGRESSION MODELS cousera
###############################################################################
from sklearn import linear_model

#Spliting
msk = np.random.rand(len(data)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#build linear model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)

#The coefficients of the model
coef= regr.coef_
interc= regr.intercept_

#######################
#Evaluation
########################
#Prediction based on the params
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])

x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))



###############################################################################
#Polynomial Regression
###############################################################################
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])


poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x) #**fit_transform** takes our x values, and output a list of our data raised from power of 0 to power of 2 (since we set the degree of our polynomial to 2).

#build model
polyreg = linear_model.LinearRegression()
train_y_ = polyreg.fit(train_x_poly, train_y)
# The coefficients

coef= polyreg.coef_
interc= polyreg.intercept_

#plot
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")


#########################
#Evaluation
#########################
test_x_poly = poly.fit_transform(test_x)
test_y_ = polyreg.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )