#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 01:34:32 2019

@author: rajdua
"""


from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = train
x = x.drop(columns  = 'Id')

y = x['Response']

x = x.fillna(0)

x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=30)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)
              # columns = ['principal component 1', 'principal component 2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'])

finalDf = pd.concat([principalDf, y], axis = 1)


# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# targets = [1, 2,3,4,5,6,7,8]
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['Response'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()

pca.explained_variance_ratio_


training = train
training = training.drop(columns  = ['Id', 'Response'])
training = training.fillna(training.median())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(training)
training = scaler.transform(training)

pca = PCA(.95)
pca.fit(training)
pca.n_components_




