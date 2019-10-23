#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 02:55:46 2019

@author: rajdua
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras import backend as K



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

y_train_enc = np.zeros((y_train.size, y_train.max()))
y_train_enc[np.arange(y_train.size), y_train - 1] = 1
y_test_enc = np.zeros((y_test.size, y_test.max()))
y_test_enc[np.arange(y_test.size), y_test - 1] = 1

model = Sequential()
model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'))
model.add(Dense((X_train.shape[1] + 8) // 2, activation='relu'))
model.add(Dense(8, activation='softmax'))

opt = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X_train, y_train_enc, epochs=42, batch_size=64, verbose=1)

_, accuracy = model.evaluate(X_train, y_train_enc)
print('Train accuracy: %.2f' % (accuracy * 100))

_, accuracy = model.evaluate(X_test, y_test_enc)
print('Overall test accuracy: %.2f' % (accuracy * 100))


X_pred  = scaler.transform(X)
predictions = model.predict_classes(X_pred)

# Convert back from 0-indexed to risk classes (1-indexed).
predictions += 1

train2 = train
train2['Response'] = predictions
predTrain = train2[['Id', 'Response']]
predTrain.set_index('Id', inplace = True)
predTrain.to_csv('NN.csv', float_format='%.0f')

# print(1 - ((1-a)* (predictions.size - 1) / (predictions.size - 97 - 1)))

predictions_test = model.predict_classes(X_test)
predictions_test += 1
# print(1 - ((1-a)* (predictions_test.size - 1) / (predictions_test.size - 97 - 1)))



from sklearn.metrics import classification_report, confusion_matrix
CM = confusion_matrix(y_test, predictions_test)
pd.DataFrame(CM).to_csv("NN(CM).csv")
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

predictions_test = model.predict_classes(test_noID)
predictions_test += 1
        
test['Response'] = predictions_test

submission = test[['Id', 'Response']]
submission.set_index('Id', inplace = True)
submission.to_csv('Sub_NN.csv', float_format='%.0f')
