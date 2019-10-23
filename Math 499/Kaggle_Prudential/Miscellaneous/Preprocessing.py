#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:10:48 2019

@author: rajdua
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('train.csv')

df['Product_Info_1'] = pd.Categorical(df['Product_Info_1'])
dfDummies = pd.get_dummies(df['Product_Info_1'], prefix = 'P1')
df = pd.concat([df, dfDummies], axis=1)

df['Product_Info_2'] = pd.Categorical(df['Product_Info_2'])
dfDummies = pd.get_dummies(df['Product_Info_2'], prefix = 'P2')
df = pd.concat([df, dfDummies], axis=1)

df['Product_Info_3'] = pd.Categorical(df['Product_Info_3'])
dfDummies = pd.get_dummies(df['Product_Info_3'], prefix = 'P3')
df = pd.concat([df, dfDummies], axis=1)

df['Product_Info_5'] = pd.Categorical(df['Product_Info_5'])
dfDummies = pd.get_dummies(df['Product_Info_5'], prefix = 'P5')
df = pd.concat([df, dfDummies], axis=1)

df['Product_Info_6'] = pd.Categorical(df['Product_Info_6'])
dfDummies = pd.get_dummies(df['Product_Info_6'], prefix = 'P6')
df = pd.concat([df, dfDummies], axis=1)

df['Product_Info_7'] = pd.Categorical(df['Product_Info_7'])
dfDummies = pd.get_dummies(df['Product_Info_7'], prefix = 'P7')
df = pd.concat([df, dfDummies], axis=1)


df['Employment_Info_2'] = pd.Categorical(df['Employment_Info_2'])
dfDummies = pd.get_dummies(df['Employment_Info_2'], prefix = 'E2')
df = pd.concat([df, dfDummies], axis=1)

df['Employment_Info_3'] = pd.Categorical(df['Employment_Info_3'])
dfDummies = pd.get_dummies(df['Product_Info_7'], prefix = 'E3')
df = pd.concat([df, dfDummies], axis=1)

df['Employment_Info_5'] = pd.Categorical(df['Employment_Info_5'])
dfDummies = pd.get_dummies(df['Product_Info_7'], prefix = 'E5')
df = pd.concat([df, dfDummies], axis=1)


df['InsuredInfo_1'] = pd.Categorical(df['InsuredInfo_1'])
dfDummies = pd.get_dummies(df['InsuredInfo_1'], prefix = 'I1')
df = pd.concat([df, dfDummies], axis=1)

df['InsuredInfo_2'] = pd.Categorical(df['InsuredInfo_2'])
dfDummies = pd.get_dummies(df['InsuredInfo_2'], prefix = 'I2')
df = pd.concat([df, dfDummies], axis=1)

df['InsuredInfo_3'] = pd.Categorical(df['InsuredInfo_3'])
dfDummies = pd.get_dummies(df['InsuredInfo_3'], prefix = 'I3')
df = pd.concat([df, dfDummies], axis=1)

df['InsuredInfo_4'] = pd.Categorical(df['InsuredInfo_4'])
dfDummies = pd.get_dummies(df['InsuredInfo_4'], prefix = 'I4')
df = pd.concat([df, dfDummies], axis=1)

df['InsuredInfo_5'] = pd.Categorical(df['InsuredInfo_5'])
dfDummies = pd.get_dummies(df['InsuredInfo_5'], prefix = 'I5')
df = pd.concat([df, dfDummies], axis=1)

df['InsuredInfo_6'] = pd.Categorical(df['InsuredInfo_6'])
dfDummies = pd.get_dummies(df['InsuredInfo_6'], prefix = 'I6')
df = pd.concat([df, dfDummies], axis=1)

df['InsuredInfo_7'] = pd.Categorical(df['InsuredInfo_7'])
dfDummies = pd.get_dummies(df['InsuredInfo_7'], prefix = 'I7')
df = pd.concat([df, dfDummies], axis=1)



df['Insurance_History_1'] = pd.Categorical(df['Insurance_History_1'])
dfDummies = pd.get_dummies(df['Insurance_History_1'], prefix = 'IH1')
df = pd.concat([df, dfDummies], axis=1)

df['Insurance_History_2'] = pd.Categorical(df['Insurance_History_2'])
dfDummies = pd.get_dummies(df['Insurance_History_2'], prefix = 'IH2')
df = pd.concat([df, dfDummies], axis=1)

df['Insurance_History_3'] = pd.Categorical(df['Insurance_History_3'])
dfDummies = pd.get_dummies(df['Insurance_History_3'], prefix = 'IH3')
df = pd.concat([df, dfDummies], axis=1)

df['Insurance_History_4'] = pd.Categorical(df['Insurance_History_4'])
dfDummies = pd.get_dummies(df['Insurance_History_4'], prefix = 'IH4')
df = pd.concat([df, dfDummies], axis=1)

df['Insurance_History_7'] = pd.Categorical(df['Insurance_History_7'])
dfDummies = pd.get_dummies(df['Insurance_History_7'], prefix = 'IH7')
df = pd.concat([df, dfDummies], axis=1)

df['Insurance_History_8'] = pd.Categorical(df['Insurance_History_8'])
dfDummies = pd.get_dummies(df['Insurance_History_8'], prefix = 'IH8')
df = pd.concat([df, dfDummies], axis=1)

df['Insurance_History_8'] = pd.Categorical(df['Insurance_History_9'])
dfDummies = pd.get_dummies(df['Insurance_History_9'], prefix = 'IH9')
df = pd.concat([df, dfDummies], axis=1)


df['Family_Hist_1'] = pd.Categorical(df['Family_Hist_1'])
dfDummies = pd.get_dummies(df['Family_Hist_1'], prefix = 'FH1')
df = pd.concat([df, dfDummies], axis=1)

df = df.drop(columns = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 
                        'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 
                        'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 
                        'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 
                        'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 
                        'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 
                        'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 
                        'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1'])



df = df.drop(columns = ['Medical_History_2', 'Medical_History_3', 'Medical_History_4', 
                        'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 
                        'Medical_History_8', 'Medical_History_9', 'Medical_History_11',
                        'Medical_History_12', 'Medical_History_13','Medical_History_14', 
                        'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 
                        'Medical_History_19', 'Medical_History_20', 'Medical_History_21',
                        'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 
                        'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 
                        'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 
                        'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 
                        'Medical_History_39', 'Medical_History_40', 'Medical_History_41',
                        'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 
                        'Medical_History_24', 'Medical_History_32'])


train  = df




df = pd.read_csv('test.csv')

df['Product_Info_1'] = pd.Categorical(df['Product_Info_1'])
dfDummies = pd.get_dummies(df['Product_Info_1'], prefix = 'P1')
df = pd.concat([df, dfDummies], axis=1)

df['Product_Info_2'] = pd.Categorical(df['Product_Info_2'])
dfDummies = pd.get_dummies(df['Product_Info_2'], prefix = 'P2')
df = pd.concat([df, dfDummies], axis=1)

df['Product_Info_3'] = pd.Categorical(df['Product_Info_3'])
dfDummies = pd.get_dummies(df['Product_Info_3'], prefix = 'P3')
df = pd.concat([df, dfDummies], axis=1)



df['Product_Info_5'] = pd.Categorical(df['Product_Info_5'])
dfDummies = pd.get_dummies(df['Product_Info_5'], prefix = 'P5')
df = pd.concat([df, dfDummies], axis=1)

df['Product_Info_6'] = pd.Categorical(df['Product_Info_6'])
dfDummies = pd.get_dummies(df['Product_Info_6'], prefix = 'P6')
df = pd.concat([df, dfDummies], axis=1)

df['Product_Info_7'] = pd.Categorical(df['Product_Info_7'])
dfDummies = pd.get_dummies(df['Product_Info_7'], prefix = 'P7')
df = pd.concat([df, dfDummies], axis=1)


df['Employment_Info_2'] = pd.Categorical(df['Employment_Info_2'])
dfDummies = pd.get_dummies(df['Employment_Info_2'], prefix = 'E2')
df = pd.concat([df, dfDummies], axis=1)

df['Employment_Info_3'] = pd.Categorical(df['Employment_Info_3'])
dfDummies = pd.get_dummies(df['Product_Info_7'], prefix = 'E3')
df = pd.concat([df, dfDummies], axis=1)

df['Employment_Info_5'] = pd.Categorical(df['Employment_Info_5'])
dfDummies = pd.get_dummies(df['Product_Info_7'], prefix = 'E5')
df = pd.concat([df, dfDummies], axis=1)


df['InsuredInfo_1'] = pd.Categorical(df['InsuredInfo_1'])
dfDummies = pd.get_dummies(df['InsuredInfo_1'], prefix = 'I1')
df = pd.concat([df, dfDummies], axis=1)

df['InsuredInfo_2'] = pd.Categorical(df['InsuredInfo_2'])
dfDummies = pd.get_dummies(df['InsuredInfo_2'], prefix = 'I2')
df = pd.concat([df, dfDummies], axis=1)

df['InsuredInfo_3'] = pd.Categorical(df['InsuredInfo_3'])
dfDummies = pd.get_dummies(df['InsuredInfo_3'], prefix = 'I3')
df = pd.concat([df, dfDummies], axis=1)

df['InsuredInfo_4'] = pd.Categorical(df['InsuredInfo_4'])
dfDummies = pd.get_dummies(df['InsuredInfo_4'], prefix = 'I4')
df = pd.concat([df, dfDummies], axis=1)

df['InsuredInfo_5'] = pd.Categorical(df['InsuredInfo_5'])
dfDummies = pd.get_dummies(df['InsuredInfo_5'], prefix = 'I5')
df = pd.concat([df, dfDummies], axis=1)

df['InsuredInfo_6'] = pd.Categorical(df['InsuredInfo_6'])
dfDummies = pd.get_dummies(df['InsuredInfo_6'], prefix = 'I6')
df = pd.concat([df, dfDummies], axis=1)

df['InsuredInfo_7'] = pd.Categorical(df['InsuredInfo_7'])
dfDummies = pd.get_dummies(df['InsuredInfo_7'], prefix = 'I7')
df = pd.concat([df, dfDummies], axis=1)



df['Insurance_History_1'] = pd.Categorical(df['Insurance_History_1'])
dfDummies = pd.get_dummies(df['Insurance_History_1'], prefix = 'IH1')
df = pd.concat([df, dfDummies], axis=1)

df['Insurance_History_2'] = pd.Categorical(df['Insurance_History_2'])
dfDummies = pd.get_dummies(df['Insurance_History_2'], prefix = 'IH2')
df = pd.concat([df, dfDummies], axis=1)

df['Insurance_History_3'] = pd.Categorical(df['Insurance_History_3'])
dfDummies = pd.get_dummies(df['Insurance_History_3'], prefix = 'IH3')
df = pd.concat([df, dfDummies], axis=1)

df['Insurance_History_4'] = pd.Categorical(df['Insurance_History_4'])
dfDummies = pd.get_dummies(df['Insurance_History_4'], prefix = 'IH4')
df = pd.concat([df, dfDummies], axis=1)

df['Insurance_History_7'] = pd.Categorical(df['Insurance_History_7'])
dfDummies = pd.get_dummies(df['Insurance_History_7'], prefix = 'IH7')
df = pd.concat([df, dfDummies], axis=1)

df['Insurance_History_8'] = pd.Categorical(df['Insurance_History_8'])
dfDummies = pd.get_dummies(df['Insurance_History_8'], prefix = 'IH8')
df = pd.concat([df, dfDummies], axis=1)

df['Insurance_History_8'] = pd.Categorical(df['Insurance_History_9'])
dfDummies = pd.get_dummies(df['Insurance_History_9'], prefix = 'IH9')
df = pd.concat([df, dfDummies], axis=1)


df['Family_Hist_1'] = pd.Categorical(df['Family_Hist_1'])
dfDummies = pd.get_dummies(df['Family_Hist_1'], prefix = 'FH1')
df = pd.concat([df, dfDummies], axis=1)

df = df.drop(columns = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 
                        'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 
                        'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 
                        'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 
                        'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 
                        'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 
                        'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 
                        'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1'])



df = df.drop(columns = ['Medical_History_2', 'Medical_History_3', 'Medical_History_4', 
                        'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 
                        'Medical_History_8', 'Medical_History_9', 'Medical_History_11',
                        'Medical_History_12', 'Medical_History_13','Medical_History_14', 
                        'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 
                        'Medical_History_19', 'Medical_History_20', 'Medical_History_21',
                        'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 
                        'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 
                        'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 
                        'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 
                        'Medical_History_39', 'Medical_History_40', 'Medical_History_41',
                        'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 
                        'Medical_History_24', 'Medical_History_32'])

test = df

New = train.merge(test, how = 'outer')

size = train.shape[0]
train = New[0:size]

test = New[size:]





