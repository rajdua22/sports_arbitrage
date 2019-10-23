import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

continuous = 'Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5'.split(', ')
discrete = 'Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32'.split(', ')

all_numerical = [*continuous, *discrete]

personal = ['Ins_Age', 'Ht', 'Wt', 'BMI']
med_hist = [s for s in all_numerical if s.startswith('Medical_History')]
fam_hist = [s for s in all_numerical if s.startswith('Family_Hist')]

# features = [*personal_features, *med_hist_features, *fam_hist_features, *prod_info_features]
features = all_numerical


def main():
    df = pd.read_csv('train.csv', delimiter=',')

    kfold = KFold(10, True)
    for train_idx, test_idx in kfold.split(df):
        df_train = df.iloc[train_idx]
        df_train = df_train.fillna(df_train.mean())

        x_train = df_train[features]
        y_train = df_train[['Response']]

        df_test = df.iloc[test_idx]
        df_test = df_test.fillna(df_test.mean())

        x_test = df_test[features]
        y_test = df_test[['Response']]

        reg = LinearRegression().fit(x_train, y_train)

        # from sklearn.feature_selection import SelectFromModel
        # model = SelectFromModel(reg, prefit=True, threshold=0)
        #
        # print(x_train.shape)
        # x_train = model.transform(x_train)
        # x_test = model.transform(x_test)
        # print(x_train.shape)
        #
        # reg = LinearRegression().fit(x_train, y_train)

        coef = list(zip(features, reg.coef_[0]))
        coef.sort(key=lambda x: x[1], reverse=True)

        top_5_pos = coef[:5]
        top_5_neg = coef[-5:]

        print('Top 5 Pos: \n', top_5_pos)
        print('Top 5 Neg: \n', top_5_neg)

        print(reg.score(x_train, y_train))

        y_pred = reg.predict(x_test)
        print("Mean squared error: %.2f"
              % mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        n = x_test.shape[0]
        k = x_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        print('R2 score: %.2f' % r2)
        print('Adjusted R2 score: %.2f' % adj_r2)

        # plt.scatter(x_test, y_test, color='black')
        # plt.plot(x_test, y_pred, color='blue', linewidth=3)
        #
        # plt.show()

        break


if __name__ == '__main__':
    main()
