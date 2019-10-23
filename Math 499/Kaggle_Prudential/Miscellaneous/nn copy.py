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

print(K.tensorflow_backend._get_available_gpus())


categorical = 'Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41'.split(', ')
continuous = 'Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5'.split(', ')
discrete = 'Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32'.split(', ')


def main():
    df = pd.read_csv('train.csv', delimiter=',')
    df = df.fillna(-1)
    df = pd.get_dummies(df, prefix=categorical, columns=categorical)
    features = list(df.columns.values)
    features.remove('Id')

    df_test_real = pd.read_csv('test.csv', delimiter=',')
    df_test_real = df_test_real.fillna(-1)
    df_test_real = pd.get_dummies(df_test_real, prefix=categorical, columns=categorical)

    features = [f for f in features if f in df_test_real and f.startswith('Medical_History')]
    # features = ['Ht', 'Wt', 'Ins_Age', 'BMI']

    kfold = KFold(10, True)
    for train_idx, test_idx in kfold.split(df):
        df_train = df.iloc[train_idx]

        x_train = df_train[features].values
        y_train = df_train[['Response']].values.flatten()
        # print(y_train)
        y_train_enc = np.zeros((y_train.size, y_train.max()))
        y_train_enc[np.arange(y_train.size), y_train - 1] = 1
        # print(y_train_enc)

        df_test = df.iloc[test_idx]

        x_test = df_test[features].values
        y_test = df_test[['Response']].values.flatten()
        y_test_enc = np.zeros((y_test.size, y_test.max()))
        y_test_enc[np.arange(y_test.size), y_test - 1] = 1

        model = Sequential()
        model.add(Dense(len(features), input_dim=len(features), activation='relu'))
        model.add(Dense((len(features) + 8) // 2, activation='relu'))
        model.add(Dense(8, activation='softmax'))

        opt = Adam(lr=0.001)

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        model.fit(x_train, y_train_enc, epochs=30, batch_size=64, verbose=1)

        _, accuracy = model.evaluate(x_train, y_train_enc)
        print('Train accuracy: %.2f' % (accuracy * 100))

        _, accuracy = model.evaluate(x_test, y_test_enc)
        print('Overall test accuracy: %.2f' % (accuracy * 100))

        for risk_class in range(1, 9):
            df_test_i = df_test.loc[df_test['Response'] == risk_class]
            x_test_i = df_test_i[features].values
            y_test_i = df_test_i[['Response']].values.flatten()
            y_test_enc_i = np.zeros((y_test_i.size, 8))
            y_test_enc_i[np.arange(y_test_i.size), y_test_i - 1] = 1

            _, accuracy = model.evaluate(x_test_i, y_test_enc_i)
            print('Risk class %d test accuracy: %.2f' % (risk_class, accuracy * 100))

        x_test_real = df_test_real[features].values
        print(x_test.shape)
        y_pred = model.predict_classes(x_test_real)
        print(y_pred)

        # Convert back from 0-indexed to risk classes (1-indexed).
        y_pred += 1

        print(len(y_pred))

        for p in y_pred:
            if not 1 <= p <= 8:
                print(p)
                assert False

        id_pred = list(zip(df_test_real[['Id']].values.flatten(), y_pred))

        print('Predict:')
        print(id_pred)

        with open('pred.csv', mode='w') as f:
            f.write('Id,Response\n')
            f.write('\n'.join(['%d,%d' % (s_id, pred) for s_id, pred in id_pred]))

        return


if __name__ == '__main__':
    main()
