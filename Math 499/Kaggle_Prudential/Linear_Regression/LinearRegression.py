#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:03:55 2019

@author: rajdua
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

X = train
X = X.drop(columns  = 'Id')

y = X['Response']

# X = X.fillna(0)
X = X.fillna(-1)

X = X.drop(columns = 'Response')

# X = X[['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 
#     'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 
#     'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 
#     'Family_Hist_4', 'Family_Hist_5']]

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=0)

scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# X_train = scaler.transform(X)

reg = LinearRegression().fit(X_train, y_train.ravel())

reg.fit(X_train, y_train.ravel())


X_pred  = scaler.transform(X)
predictions = reg.predict(X_pred)
predictions = np.around(predictions)

size = predictions.size

for a in range(0,size):
    if predictions[a] > 8:
        predictions[a] = 8
    if predictions[a] < 1:
        predictions[a] = 1
        
print('Accuracy of linear regression classifier on entire dataset:')
print(r2_score(predictions, y))
    
train2 = train
train2['Response'] = predictions
predTrain = train2[['Id', 'Response']]
predTrain.set_index('Id', inplace = True)
predTrain.to_csv('Linear_Regression.csv', float_format='%.0f')

predictions_test = reg.predict(X_test)
predictions_test = np.around(predictions_test)

size = predictions_test.size

for a in range(0,size):
    if predictions_test[a] > 8:
        predictions_test[a] = 8
    if predictions_test[a] < 1:
        predictions_test[a] = 1


print('Accuracy of linear regression classifier on test set:')
print(r2_score(predictions_test, y_test))

a = reg.coef_

coef_df = pd.DataFrame(np.transpose(a))
coef_df['names'] = X.columns
# coef_df.to_csv('All_Var(P2_spread).csv')


from sklearn.metrics import classification_report, confusion_matrix
CM = confusion_matrix(y_test, predictions_test)
pd.DataFrame(CM).to_csv("Linear_REgression_(CM).csv")
print(classification_report(y_test, predictions_test))


test_noID = test.drop(columns = ['Id'])
# test_noID = test_noID.fillna(0)
test_noID = test_noID.fillna(-1)
# test_noID = test_noID.drop(columns = 'Response')

# test_noID = test_noID[['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 
#     'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 
#     'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 
#     'Family_Hist_4', 'Family_Hist_5']]

scaler = StandardScaler()
scaler.fit(test_noID)
test_scaled = scaler.transform(test_noID)

predictions_test = reg.predict(test_scaled)
predictions_test = np.around(predictions_test)

size = predictions_test.size

for a in range(0,size):
    if predictions_test[a] > 8:
        predictions_test[a] = 8
    if predictions_test[a] < 1:
        predictions_test[a] = 1
        
test['Response'] = predictions_test

submission = test[['Id', 'Response']]
submission.set_index('Id', inplace = True)
submission.to_csv('Sub_Linear_Regression.csv', float_format='%.0f')

