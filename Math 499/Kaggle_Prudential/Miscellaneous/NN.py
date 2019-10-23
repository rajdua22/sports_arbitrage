#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 23:11:18 2019

@author: rajdua
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

X = train
X = X.drop(columns  = 'Id')

y = X['Response']

y = pd.Categorical(y)
y = pd.get_dummies(y)

X = X.fillna(0)

X = X[['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 
    'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 
    'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 
    'Family_Hist_4', 'Family_Hist_5']]



X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=0)
scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test2 = X_test
X_test = scaler.transform(X_test)


def baseline_model():
    model = Sequential()
    model.add(Dense(64, input_dim=13, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=5, verbose=1)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# model = Sequential()
# model.add(Dense(64, input_dim=219, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# model.fit(X_train, y_train,
#           epochs=30,
#           batch_size=128)

print('Accuracy of logistic regression classifier on test set:')
print(model.evaluate(X_test, y_test, batch_size = 128))

X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)

print('Accuracy of logistic regression classifier on entire dataset:')
print(model.evaluate(X_scaled, y, batch_size = 128))