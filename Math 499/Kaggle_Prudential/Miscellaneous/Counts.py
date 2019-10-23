#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 09:32:03 2019

@author: rajdua
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

train = pd.read_csv('Documents/Math 499/Logistic_Regression/Logistic_Regression.csv')
actual = train['Response']
actual = actual.value_counts()
actual.to_csv('Documents/Math 499/Logistic_Regression/Logistic_Regression_Counts.csv')


train = pd.read_csv('Documents/Math 499/Linear_Regression/Linear_Regression.csv')
actual = train['Response']
actual = actual.value_counts()
actual.to_csv('Documents/Math 499/Linear_Regression/Linear_Regression_Counts.csv')

train = pd.read_csv('Documents/Math 499/Random_Forest/RandomF_orest.csv')
actual = train['Response']
actual = actual.value_counts()
actual.to_csv('Documents/Math 499/Random_Forest/Random_Forest_Counts.csv')

train = pd.read_csv('Documents/Math 499/Naive_Bayes/Naive_Bayes.csv')
actual = train['Response']
actual = actual.value_counts()
actual.to_csv('Documents/Math 499/Naive_Bayes/Naive_Bayes_Counts.csv')

train = pd.read_csv('Documents/Math 499/ADA/ADA.csv')
actual = train['Response']
actual = actual.value_counts()
actual.to_csv('Documents/Math 499/ADA/ADA_counts.csv')

train = pd.read_csv('Documents/Math 499/Decision_Trees/Decision_Trees.csv')
actual = train['Response']
actual = actual.value_counts()
actual.to_csv('Documents/Math 499/Decision_Trees/Decision_Trees_Counts.csv')

train = pd.read_csv('Documents/Math 499/Extra_Trees/Extra_Trees.csv')
actual = train['Response']
actual = actual.value_counts()
actual.to_csv('Documents/Math 499/Extra_Trees/Extra_Trees_Counts.csv')


train = pd.read_csv('Documents/Math 499/E_LR_RF_NB_ADA/E_LR__RF_NB_ADA.csv')
actual = train['Response']
actual = actual.value_counts()
actual.to_csv('Documents/Math 499/E_LR_RF_NB_ADA/E_LR_RF_NB_ADA_Counts.csv')

train = pd.read_csv('Documents/Math 499/NN/NN.csv')
actual = train['Response']
actual = actual.value_counts()
actual.to_csv('Documents/Math 499/NN/NN_Counts.csv')

train = pd.read_csv('Smote_ALL.csv')
actual = train['Response']
actual = actual.value_counts()
actual.to_csv('Documents/Math 499/NN/Smote_Counts.csv')

train = pd.read_csv('XGB.csv')
actual = train['Response']
actual = actual.value_counts()
actual.to_csv('XGB_Boost_Counts.csv')


train = pd.read_csv('E5.csv')
actual = train['Response']
actual = actual.value_counts()
actual.to_csv('E5_Counts.csv')

