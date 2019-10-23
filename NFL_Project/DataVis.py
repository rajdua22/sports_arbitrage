#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 01:28:51 2019

@author: rajdua
"""


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import RFE
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV as CCV
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV as CCV
from sklearn import preprocessing

betting = pd.read_csv("nfl-scores-and-betting-data/spreadspoke_scores.csv")
qb = pd.read_csv("nflstatistics/Game_Logs_Quarterback.csv")
betting['schedule_date'] = pd.to_datetime(betting['schedule_date'])


betting.head()
betting.isnull().sum()
betting = betting.dropna(subset  = ['spread_favorite', 'over_under_line'])
plt.figure(0)
plt.hist(betting.spread_favorite, alpha=0.5)
sns.rugplot(betting.spread_favorite)

plt.figure(1)
plt.hist(betting.schedule_date, alpha= 0.5)


teams = pd.read_csv("nfl-scores-and-betting-data/nfl_teams.csv")
teams = teams[["team_name", "team_id"]]
team_dict = dict(zip(teams.team_name, teams.team_id))
betting = betting.replace({"team_home" : team_dict})
betting = betting.replace({"team_away" : team_dict})

def favored(x):
    if (x['team_home'] == x['team_favorite_id']):
        return True
    else:
        return False
    
betting['spread_favorite'] = betting['spread_favorite'] * -1
    
betting['home_favorite'] = betting.apply(lambda x: favored(x), axis = 1)
betting = betting[betting["over_under_line"] != ' ']
betting.over_under_line = pd.to_numeric(betting.over_under_line)
betting['over_hit'] = (betting.score_home + betting.score_away > betting.over_under_line)

betting['score_spread'] = betting['score_home']-betting['score_away']


def covered(x):
    if (x['home_favorite']):
        if (x['score_spread'] > x['spread_favorite']):
            return True
        else:
            return False
    else:
        if ((-1*x['score_spread']) > x['spread_favorite']):
            return True
        else:
            return False
        
betting['covered_spread'] = betting.apply(lambda x: covered(x), axis = 1)


cover = betting.covered_spread.sum()
total = betting.covered_spread.count()
percentCoveredSpread = cover / total

plt.figure(2)
sns.set(style="darkgrid")
sns.relplot(x="over_under_line", y ="spread_favorite", data = betting)

qb.Year = qb.Year.apply(lambda x: x % 100)
qb.Year = qb.Year.apply(lambda x: format(x, '02d'))
qb.Year = qb.Year.apply(lambda x: str(x))

qb["newDate"] = qb["Game Date"] + "/" + qb.Year.map(str)
qb["newDate"] = pd.to_datetime(qb["newDate"])

qb = qb[qb["Games Started"] == "1"]

plt.figure(3)
plt.hist(qb.newDate, alpha=0.5)

qb = qb[qb["TD Passes"] != '--']
qb = qb[qb['Passing Yards'] != '--']
qb = qb[qb['Passes Completed'] != '--']

qb['TD Passes'] = pd.to_numeric(qb['TD Passes'])
qb['Passes Attempted'] = pd.to_numeric(qb['Passes Attempted'])
qb['Passing Yards'] = pd.to_numeric(qb['Passing Yards'])


plt.figure(4)
plt.hist(qb['TD Passes'], alpha=0.5)

plt.figure(5)
sns.relplot(x="Passes Attempted", y ="Passing Yards", data = qb)

plt.figure(6)
sns.relplot(x="TD Passes", y ="Passing Yards", data = qb)

final = pd.merge(qb, betting, left_on = ['newDate', 'Opponent'], 
                 right_on = ['schedule_date', 'team_away'], how = 'inner')

final2 = pd.merge(qb, final, left_on = ['newDate', 'Opponent'], right_on = ['schedule_date', 'team_home'])
plt.figure(7)
plt.hist(final2['schedule_season'], alpha=0.5)

df = final2.drop(columns  = ["Position_x", "Year_x", "Game Date_x", "Games Played_x", "Games Started_x", "newDate_x",
                                 "Position_y", "Year_y", "Game Date_y", "Games Played_y", "Games Started_y", "newDate_y"])
    
plt.figure(8)
plt.hist(df['spread_favorite'], alpha=0.5)

def FavoriteQBYards(x):
    if (x['home_favorite']):
        return x['Passing Yards_y']
    else:
        return x['Passing Yards_x']
    
df['favorite_passing_yards'] = df.apply(lambda x: FavoriteQBYards(x), axis = 1)

def UnderdogQBYards(x):
    if (x['home_favorite']):
        return x['Passing Yards_x']
    else:
        return x['Passing Yards_y']
    
df['underdog_passing_yards'] = df.apply(lambda x: UnderdogQBYards(x), axis = 1)

def FavoriteQBTD(x):
    if (x['home_favorite']):
        return x['TD Passes_y']
    else:
        return x['TD Passes_x']
    
df['favorite_TD_passes'] = df.apply(lambda x: FavoriteQBTD(x), axis = 1)

def UnderdogQBTD(x):
    if (x['home_favorite']):
        return x['TD Passes_x']
    else:
        return x['TD Passes_y']
    
df['underdog_TD_passes'] = df.apply(lambda x: UnderdogQBTD(x), axis = 1)

plt.figure(9)
a = df.groupby(df['spread_favorite'])['underdog_TD_passes'].mean()
plt.scatter(a.index, a)

plt.figure(10)
a = df.groupby(df['spread_favorite'])['favorite_TD_passes'].mean()
plt.scatter(a.index, a)

plt.figure(11)
a = df.groupby(df['spread_favorite'])['underdog_passing_yards'].mean()
plt.scatter(a.index, a)

plt.figure(12)
a = df.groupby(df['spread_favorite'])['favorite_passing_yards'].mean()
plt.scatter(a.index, a)


reduced_df = df[['underdog_TD_passes', 'favorite_TD_passes', 'underdog_passing_yards', 
                  'favorite_passing_yards', 'home_favorite', 'team_home', 'team_away',
                  'score_home', 'score_away', 'team_away', 'team_favorite_id', 'spread_favorite',
                  'over_under_line', 'score_spread', 'covered_spread', 'over_hit']]

train_spread = reduced_df[['underdog_TD_passes', 'favorite_TD_passes', 'underdog_passing_yards', 
                          'favorite_passing_yards', 'spread_favorite', 'covered_spread']]


train_over = reduced_df[['underdog_TD_passes', 'favorite_TD_passes', 'underdog_passing_yards', 
                        'favorite_passing_yards', 'over_under_line', 'over_hit']]

X_spread = train_spread.drop('covered_spread', axis=1)
y_spread = train_spread['covered_spread']
X_over = train_over.drop('over_hit',axis=1)
y_over = train_over['over_hit']
base = LDA()


# choose 7 best features
rfe = RFE(base, 5)
rfe = rfe.fit(X_over, y_over)

# features
print(rfe.support_)
print(rfe.ranking_)

boost = xgb.XGBClassifier()
X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(
    X_over, y_over, test_size=0.33, random_state=42)
boost.fit(X_train_over, y_train_over)

preds = boost.predict(X_test_over)

print(roc_auc_score(y_test_over, preds))

print(accuracy_score(y_test_over, preds, normalize=True))

models = []

models.append(('LRG', LogisticRegression(solver='liblinear')))
models.append(('KNB', KNeighborsClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('XGB', xgb.XGBClassifier(random_state=0)))
models.append(('RFC', RandomForestClassifier(random_state=0, n_estimators=100)))
models.append(('DTC', DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=5)))

# evaluate each model by average and standard deviations of roc auc
results = []
names = []

for name, m in models:
    kfold = model_selection.KFold(n_splits=5, random_state=0)
    cv_results = model_selection.cross_val_score(m, X_over, y_over, cv=kfold, scoring = 'roc_auc')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


boost = xgb.XGBClassifier()
rfc = RandomForestClassifier(random_state=0, n_estimators=100)
lrg = LogisticRegression(solver='liblinear')
vote = VotingClassifier(estimators=[('boost', boost), ('rfc', rfc), ('lrg', lrg)], voting='soft')

model = CCV(vote, method='isotonic', cv=3)
model.fit(X_train_over, y_train_over)
preds = model.predict(X_test_over)
print(accuracy_score(y_test_over, preds, normalize=True))





    
rfe = RFE(base, 5)
rfe = rfe.fit(X_spread, y_spread)

# features
print(rfe.support_)
print(rfe.ranking_)

boost = xgb.XGBClassifier()
X_train_spread, X_test_spread, y_train_spread, y_test_spread = train_test_split(
    X_spread, y_spread, test_size=0.33, random_state=42)
boost.fit(X_train_spread, y_train_spread)

preds = boost.predict(X_test_spread)

print(roc_auc_score(y_test_spread, preds))

print(accuracy_score(y_test_spread, preds, normalize=True))

models = []

models.append(('LRG', LogisticRegression(solver='liblinear')))
models.append(('KNB', KNeighborsClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('XGB', xgb.XGBClassifier(random_state=0)))
models.append(('RFC', RandomForestClassifier(random_state=0, n_estimators=100)))
models.append(('DTC', DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=5)))

# evaluate each model by average and standard deviations of roc auc
results = []
names = []

for name, m in models:
    kfold = model_selection.KFold(n_splits=5, random_state=0)
    cv_results = model_selection.cross_val_score(m, X_spread, y_spread, cv=kfold, scoring = 'roc_auc')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


boost = xgb.XGBClassifier()
rfc = RandomForestClassifier(random_state=0, n_estimators=100)
lrg = LogisticRegression(solver='liblinear')
vote = VotingClassifier(estimators=[('boost', boost), ('rfc', rfc), ('lrg', lrg)], voting='soft')

model = CCV(vote, method='isotonic', cv=3)
model.fit(X_train_spread, y_train_spread)
preds = model.predict(X_test_spread)
print(accuracy_score(y_test_spread, preds, normalize=True))


fig, ax = plt.subplots()
scatter = ax.scatter(reduced_df.spread_favorite, reduced_df.underdog_TD_passes, 
                     c = reduced_df.covered_spread, cmap="binary")
# legend1 = ax.legend(*scatter.legend_elements(),
#                     loc="lower left", title="Classes")
# ax.add_artist(legend1)


ranger = reduced_df.spread_favorite.unique()

EV_total = np.zeros(ranger.size)
j = 0
for i in ranger:
    
    grouped  = reduced_df.groupby('spread_favorite')
    spread = i
    
    # td_passes = grouped.get_group(spread)['underdog_TD_passes'].mean()
    td_passes = 1.5
    
    current  = grouped.get_group(spread)
    size = current.shape[0]
    covered_set = current[current['covered_spread']]
    size_covered_set = covered_set.shape[0]
    size_both = covered_set[covered_set['underdog_TD_passes'] > td_passes].shape[0]
    underdog_overTD = current[current['underdog_TD_passes'] > td_passes]
    size_underdog_overTD = underdog_overTD.shape[0]
    sanity_check_size = underdog_overTD[underdog_overTD['covered_spread']].shape[0]
    
    percent_covered = size_covered_set/ size
    percent_overTD = size_underdog_overTD/ size
    
    print('Covered Spread:', end = ' ') 
    print(percent_covered)
    print('Under', end = ' ')
    print(td_passes, end  = '')
    print(':', end = ' ')
    print(percent_overTD)
    
    
    # if(percent_covered <= 50):  
    #     m1 = 100*(1-percent_covered) / percent_covered
    #     a1 = (100 + m1) / 100
    # else:
    #     m1 = 100*(percent_covered) / (1-percent_covered)
    #     a1 = (100 + m1) / m1
    
    # if(percent_underTD <= 50):  
    #     m2 = 100*(1-percent_underTD) / percent_underTD
    #     a2 = (100 + m2) / 100
    # else:
    #     m2 = 100*(percent_underTD) / (1-percent_underTD)
    #     a2 = (100 + m2) / m2
    
    
    a1 = 210/110
    a2 = 300/100
    f1 = a2 / (a1 + a2)
    f2 = a1 / (a2 + a1)
    
    if (sanity_check_size != size_both):
        print(spread)
        print('error')
    
    alpha = size_both / size
    beta = (size_covered_set - size_both) / size
    gamma = (size_underdog_overTD - size_both) / size
    
    EV = alpha * (a1 * f1 + a2*f2) + beta * (a1 * f1) + gamma * (a2 * f2) - f1 - f2
    
    EV_total[j] = EV
    j = j+ 1
    
EV_graph = pd.DataFrame(ranger, EV_total)

plt.scatter(EV_graph, EV_graph.index)




ranger = reduced_df.over_under_line.unique()

EV_total = np.zeros(ranger.size)
j = 0
for i in ranger:
    
    grouped  = reduced_df.groupby('over_under_line')
    over = i
    
    # td_passes = grouped.get_group(spread)['underdog_TD_passes'].mean()
    td_passes = 1.5
    
    current  = grouped.get_group(over)
    size = current.shape[0]
    covered_set = current[current['over_hit']]
    size_covered_set = covered_set.shape[0]
    size_both = covered_set[covered_set['over_hit'] < td_passes].shape[0]
    underdog_underTD = current[current['over_hit'] < td_passes]
    size_underdog_underTD = underdog_overTD.shape[0]
    sanity_check_size = underdog_underTD[underdog_underTD['over_hit']].shape[0]
    
    percent_covered = size_covered_set/ size
    percent_underTD = size_underdog_underTD/ size
    
    # print('Covered Spread:', end = ' ') 
    # print(percent_covered)
    # print('Under', end = ' ')
    # print(td_passes, end  = '')
    # print(':', end = ' ')
    # print(percent_underTD)
    
    
    # if(percent_covered <= 50):  
    #     m1 = 100*(1-percent_covered) / percent_covered
    #     a1 = (100 + m1) / 100
    # else:
    #     m1 = 100*(percent_covered) / (1-percent_covered)
    #     a1 = (100 + m1) / m1
    
    # if(percent_underTD <= 50):  
    #     m2 = 100*(1-percent_underTD) / percent_underTD
    #     a2 = (100 + m2) / 100
    # else:
    #     m2 = 100*(percent_underTD) / (1-percent_underTD)
    #     a2 = (100 + m2) / m2
    
    a1 = 210/110
    a2 = 250/100
    f1 = a2 / (a1 + a2)
    f2 = a1 / (a2 + a1)
    
    if (sanity_check_size != size_both):
        print(spread)
        print('error')
    
    alpha = size_both / size
    beta = (size_covered_set - size_both) / size
    gamma = (size_underdog_underTD - size_both) / size
    
    EV = alpha * (a1 * f1 + a2*f2) + beta * (a1 * f1) + gamma * (a2 * f2) - f1 - f2
    
    EV_total[j] = EV
    j = j+ 1
    
EV_graph = pd.DataFrame(ranger, EV_total)

plt.scatter(EV_graph, EV_graph.index)



















    










        
    





