#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 02:18:20 2019

@author: rajdua
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('train.csv')
a = list(df.columns)
b = a[0:79]
a = a[79:127]

med  = df[a]
med = med.sum(axis = 1)

df['Product_Info_2'] = pd.Categorical(df['Product_Info_2'])
dfDummies = pd.get_dummies(df['Product_Info_2'], prefix = 'P2')
df = pd.concat([df, dfDummies], axis=1)

train = df[b]
train = train.drop(columns = 'Product_Info_2')
train['Response'] = df['Response']
train['Keyword'] = med
#train = pd.concat([train, dfDummies], axis=1)
# train = pd.concat([df, dfDummies], axis=1)


df = pd.read_csv('test.csv')

a = list(df.columns)
b = a[0:79]
a = a[79:127]

med  = df[a]
med = med.sum(axis = 1)

df['Product_Info_2'] = pd.Categorical(df['Product_Info_2'])
dfDummies = pd.get_dummies(df['Product_Info_2'], prefix = 'P2')
df = pd.concat([df, dfDummies], axis=1)

test = df[b]
test = test.drop(columns = 'Product_Info_2')
test['Keyword'] = med
# test = pd.concat([test, dfDummies], axis = 1)
# test = pd.concat([df, dfDummies], axis=1)
