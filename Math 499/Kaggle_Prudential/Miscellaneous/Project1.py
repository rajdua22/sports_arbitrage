#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:06:54 2019

@author: rajdua
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train.csv')

train['Response'].hist()
train['Response'].value_counts()

def classify(x):
    if (x< 7):
        return 0
    else:
        return 1

train['Response'] = train['Response'].apply(classify)

train['Response'].hist()
train['Response'].value_counts()

y = train['Response']
X = train.drop(columns  = 'Response')

X['Product_Info_2'] = pd.Categorical(X['Product_Info_2'])
dfDummies = pd.get_dummies(X['Product_Info_2'], prefix = 'category')
X = pd.concat([X, dfDummies], axis=1)
X = X.drop(columns = ['Id', 'Product_Info_2'])
X = X.fillna(0)

X = X[['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 
     'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 
     'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 
     'Family_Hist_4', 'Family_Hist_5']]

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

X = df.drop(columns  = 'Response')
y = df['Response']

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=0)

scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test2 = X_test
X_test = scaler.transform(X_test)
logreg = LogisticRegressionCV(penalty = 'elasticnet', solver = 'saga', max_iter = 50000, 
                              l1_ratios = [0.1, 0.1], Cs = 10, cv = 5)
logreg.fit(X_train, y_train.ravel())


print('Accuracy of logistic regression classifier on test set:')
print(logreg.score(X_test, y_test))

X_scaled = scaler.transform(X)
predictions = logreg.predict_proba(X_scaled)

print('Accuracy of logistic regression classifier on entire dataset:')
print(logreg.score(X_scaled, y))

a = logreg.coef_

coef_df = pd.DataFrame(np.transpose(a))
coef_df['names'] = X.columns

coef_df.to_csv('499(3)LR.csv')
