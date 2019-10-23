#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 02:44:55 2019

@author: rajdua
"""


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

clf1 = DecisionTreeClassifier(max_depth=20, min_samples_split=20,
     random_state=0)

X = train
X = X.drop(columns  = 'Id')

y = X['Response']

# X = X.fillna(0)
X = X.fillna(-1)

X = X.drop(columns = 'Response')


X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=0)

scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf1 = clf1.fit(X_train, y_train)
clf1.score(X_train,y_train)

X_pred  = scaler.transform(X)
predictions = clf1.predict(X_pred)

train2 = train
train2['Response'] = predictions
predTrain = train2[['Id', 'Response']]
predTrain.set_index('Id', inplace = True)
predTrain.to_csv('Decision_trees.csv', float_format='%.0f')
print('Accuracy of logistic regression classifier on entire dataset:')
a = r2_score(predictions, y)
print(a)
# print(1 - ((1-a)* (predictions.size - 1) / (predictions.size - 97 - 1)))

predictions_test = clf1.predict(X_test)
print('Accuracy of logistic regression classifier on test set:')
a = r2_score(predictions_test, y_test)
print(a)
# print(1 - ((1-a)* (predictions_test.size - 1) / (predictions_test.size - 97 - 1)))



from sklearn.metrics import classification_report, confusion_matrix
CM = confusion_matrix(y_test, predictions_test)
pd.DataFrame(CM).to_csv("Decision_trees(CM).csv")
print(classification_report(y_test, predictions_test))




test_noID = test.drop(columns = ['Id'])
# test_noID = test_noID.fillna(0)
test_noID = test_noID.fillna(-1)
# test_noID = test_noID.drop(columns = 'Response')

scaler = StandardScaler()
scaler.fit(test_noID)
test_noID = scaler.transform(test_noID)

# test_noID = test_noID[['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 
#      'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 
#      'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 
#      'Family_Hist_4', 'Family_Hist_5']]

predictions_test = clf1.predict(test_noID)
        
test['Response'] = predictions_test

submission = test[['Id', 'Response']]
submission.set_index('Id', inplace = True)
submission.to_csv('Sub_Decision_Trees.csv', float_format='%.0f')
