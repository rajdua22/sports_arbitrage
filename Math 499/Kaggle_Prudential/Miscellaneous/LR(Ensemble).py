#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 22:42:20 2019

@author: rajdua
"""


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

X = train
X = X.drop(columns  = 'Id')

y = X['Response']

# X = X.fillna(0)
X = X.fillna(X.median())

X = X.drop(columns = 'Response')

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# X = X[['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 
#     'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 
#     'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 
#     'Family_Hist_4', 'Family_Hist_5']]


clf = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                           random_state=1)

clf.fit(X, y)  
clf.score(X, y)  

s = clf.predict(X)

test_noID = test.drop(columns = ['Id'])
# test_noID = test_noID.fillna(0)
test_noID = test_noID.fillna(test_noID.median())
test_noID = test_noID.drop(columns = 'Response')

scaler = StandardScaler()
scaler.fit(test_noID)
test_noID = scaler.transform(test_noID)

# test_noID = test_noID[['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 
#      'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 
#      'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 
#      'Family_Hist_4', 'Family_Hist_5']]

predictions_test = clf.predict(test_noID)
        
test['Response'] = predictions_test

submission = test[['Id', 'Response']]
submission.set_index('Id', inplace = True)
submission.to_csv('Logistic__Regression.csv', float_format='%.0f')



