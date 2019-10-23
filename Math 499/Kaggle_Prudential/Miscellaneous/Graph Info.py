#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:21:03 2019

@author: rajdua
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv')
    
actual = train['Response']

actual = actual.value_counts()

actual.to_csv('ActualResponses.csv')


actual = pd.read_csv('AllVar.csv')
actual = actual['Response']
actual = actual.value_counts()
actual.to_csv('AllVarResponses.csv')


actual = pd.read_csv('NumVar.csv')
actual = actual['Response']
actual = actual.value_counts()
actual.to_csv('NumVarResponses.csv')


actual = pd.read_csv('TrainPred_All_Var(.csv')
actual = actual['Response']
actual = actual.value_counts()
actual.to_csv('ALLVar(P2Spread)Responses.csv')


actual = pd.read_csv('TrainPred_All(2).csv')
actual = actual['Response']
actual = actual.value_counts()
actual.to_csv('ALLVar(2)Responses.csv')

actual = pd.read_csv('TrainPred_All(3).csv')
actual = actual['Response']
actual = actual.value_counts()
actual.to_csv('ALLVar(3)Responses.csv')



