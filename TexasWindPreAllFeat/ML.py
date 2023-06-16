# example of making predictions for a regression problem
from keras.models import Sequential
from keras.layers import Dense
import keras
from tensorflow import random
from sklearn.linear_model import LinearRegression
import pandas as pd
from keras.layers import BatchNormalization,Dropout,Activation
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from scipy import stats
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from  TexasData import *
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm

def TrainML(dataset,layer_count,neuron_count):
    X_train, y_train,X_val,y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()

    random.set_seed(1)
    activations=["sigmoid","relu","tanh"] # "sigmoid","relu","tanh"
    modelFolder = "./models_" + str(dataset.train_year) + "/ML"
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)

    for activation in activations:
        model = Sequential()
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
        config=activation
        model.add(Dense(neuron_count, input_dim=X_train.shape[1], activation=activation)) #input_shape=((X_train.shape[0],X_train.shape[1]-1)),
        for i in range(layer_count-1):
            model.add(Dense(neuron_count, activation=activation))
            # model.add(Dropout(0.5))
            # model.add(Dense(neuron_count))
            # model.add(BatchNormalization())
            # model.add(Activation(activation))
        # model.add(Dense(neuron_count))
        config=config+"_"+str(layer_count)+"_"+str(neuron_count)
        model.add(Dense(1, activation=activation))
        # model.add(BatchNormalization())
        print("*************Model Summary**************")
        print(model.summary())
        print("***************************")

        model.compile(loss='mse', optimizer='Adam',metrics=['mse','mae'])
        model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=3000, verbose=1,batch_size=1024,callbacks=callback)#
        model.save(modelFolder+'/model_'+config)

def TestML(dataset,layer_count,neuron_count):
    X_train, y_train,X_val,y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
    results = pd.DataFrame(columns=["config", "r2", "mse", "mae"])
    """Save model and load"""
    activations=["sigmoid"]
    modelFolder = "./models_" + str(dataset.train_year) + "/ML"

    for activation in activations:
        config = activation + "_" + str(layer_count)+"_"+str(neuron_count)
        model = keras.models.load_model(modelFolder+'/model_'+config)

        print("***  "+activation +" "+config)
        # print(model.summary())
        # print("***************************")
        preds = model.predict(X_test)
        r2,mse,mae=dataset.EvaluateResults(modelFolder, X_test, y_test, preds)

        results = results.append({"config": config, "r2": r2, "mse": mse, "mae": mae}, ignore_index=True)
    return results.copy()

def TestlMLEachYear(dataset, model_path):
    X_train, y_train,X_val,y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()

    modelFolder = "./models_" + str(dataset.train_year) + "/ML/" +model_path
    model = keras.models.load_model(modelFolder)

    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)

    r2_dic = {}
    mae_dic = {}
    mse_dic = {}
    for year in range(2003, 2014, 1):
        if year == dataset.train_year:
            x_data = X_train
            y_data = y_train
        else:
            x_data, y_data = dataset.GetEra5_Texas_TestData(year)

        y_pred = model.predict(x_data)
        # y_pred = model.predict(np.delete(x_data,22,axis=1))
        r2, mse, mae = dataset.EvaluateResults(modelFolder, x_data, y_data, y_pred)
        r2_dic[year] = r2
        mae_dic[year] = mae
        mse_dic[year] = mse
    return r2_dic, mae_dic, mse_dic

def TestMLYearly(INPUT_PATH,model_path):
    # dictionaries for errors
    cols = range(2003, 2014, 1)
    r2 = pd.DataFrame(columns=cols)
    mae = pd.DataFrame(columns=cols)
    mse = pd.DataFrame(columns=cols)

    # RefactorReesData()
    for year in range(2003, 2014, 1):
        print("*********************")
        print(year)
        dataset = DataSet(INPUT_PATH, train_year=year,  feature_sel=0)

        r2_dic, mae_dic, mse_dic = TestlMLEachYear(dataset, model_path)
        r2 = r2.append(pd.Series(r2_dic, index=r2_dic.keys()), ignore_index=True)
        mae = mae.append(pd.Series(mae_dic, index=mae_dic.keys()), ignore_index=True)
        mse = mse.append(pd.Series(mse_dic, index=mse_dic.keys()), ignore_index=True)

    if not os.path.exists("./Results/"):
        os.makedirs("./Results/")

    r2.to_csv("./Results/ML_"+model_path+"_r2.csv")
    mae.to_csv("./Results/ML_"+model_path+"_mae.csv")
    mse.to_csv("./Results/ML_"+model_path+"_mse.csv")

    """ calculate summary tables"""
    columns = ["ML_r2", "ML_mae", "ML_mse"]
    summary_table = pd.DataFrame(columns=columns)

    for year_t in cols:
        temp = []
        index = year_t - cols[0]
        # xgboost_r2_mean = ((xgboost_r2.sum(axis=1))[index] - xgboost_r2.iloc[index, index]) / (len(cols) - 1)
        # xgboost_r2_mean = ((xgboost_r2.sum(axis=1))[index] - xgboost_r2.iloc[index, index]) / (len(cols) - 1)
        temp.append(((r2.sum(axis=1))[index] - r2.iloc[index, index]) / (len(cols) - 1))
        temp.append(((mae.sum(axis=1))[index] - mae.iloc[index, index]) / (len(cols) - 1))
        temp.append(((mse.sum(axis=1))[index] - mse.iloc[index, index]) / (len(cols) - 1))

        # summary_table=summary_table.append(np.reshape(temp,(1,-1)))
        summary_table = summary_table.append(pd.Series(temp, index=columns), ignore_index=True)
    summary_table.to_csv("./Results/ML_"+model_path+"_summary.csv")

def TestLinear(dataset):
    X_train, y_train,X_val,y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
    """Save model and load"""

    modelFolder="./Models/"+station+"_models_"+str(dataset.train_year)+"/Linear/automl.pkl"
    model = pickle.load(open(modelFolder, 'rb'))
    print("*************" + str(dataset.train_year) + "**************")
    preds = model.predict(X_test)
    dataset.EvaluateResults(modelFolder, X_test, y_test, preds)

def TrainLineargv(dataset):
    X_train, y_train, X_val, y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
    # X_train=np.delete(X_train,22,axis=1)
    # X_val=np.delete(X_val,22,axis=1)

    random.set_seed(1)
    modelFolder = "./models_" + str(dataset.train_year) + "/Linear"
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)
    model = LinearRegression()
    # model = LinearRegression()
    param_grid = {"fit_intercept": [False]}

    split_index = [-1] * len(X_train) + [0] * len(X_val)
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    pds = PredefinedSplit(test_fold=split_index)
    clf = GridSearchCV(estimator=model,param_grid=param_grid, cv=pds)

    clf.fit(X, y)

    best_model=clf.best_estimator_

    # preds = best_model.predict(X_train)
    # dataset.EvaluateResults(modelFolder, X_train, y_train, preds)
    #
    # print("coeff")
    # for i, value in enumerate(dataset.features):
    #     print(value, ":", best_model.coef_[0, i])

    with open(modelFolder + '/automl.pkl', 'wb') as f:
        pickle.dump(best_model, f, pickle.HIGHEST_PROTOCOL)

def TrainLinear(dataset):
    X_train, y_train, X_val, y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
    # X_train=np.delete(X_train,22,axis=1)
    # X_val=np.delete(X_val,22,axis=1)

    random.set_seed(1)
    modelFolder="./Models/"+station+"_models_"+str(dataset.train_year)+"/Linear"
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)
    model = LinearRegression(fit_intercept=False)

    # model.fit(X_train , y_train)
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    model.fit(X , y)

    best_model=model
    print(model.intercept_,  model.score(X_train, y_train))
    # print( model.coef_)

    with open(modelFolder + '/automl.pkl', 'wb') as f:
        pickle.dump(best_model, f, pickle.HIGHEST_PROTOCOL)

def TrainLinearStats(dataset):
    X_train, y_train, X_val, y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
    # X_train=np.delete(X_train,22,axis=1)
    # X_val=np.delete(X_val,22,axis=1)
    random.set_seed(1)
    modelFolder="./Models/"+station+"_models_"+str(dataset.train_year)+"/Linear"
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)

    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    # clf = GridSearchCV(estimator=model,param_grid=param_grid, cv=pds)

    # clf.fit(X, y)
    # best_model=clf.best_estimator_
    # X = sm.add_constant(X)
    best_model =sm.OLS(y, X).fit()
    # best_model =sm.OLS(y_train,X_train).fit()


    # preds = best_model.predict(X_train)
    # dataset.EvaluateResults(modelFolder, X_train, y_train, preds)
    #
    # print("coeff")
    # for i, value in enumerate(dataset.features):
    #     print(value, ":", best_model.coef_[0, i])

    with open(modelFolder + '/automl.pkl', 'wb') as f:
        pickle.dump(best_model, f, pickle.HIGHEST_PROTOCOL)

def TestLinearYearly(INPUT_PATH):
    start_year=2003 #2003
    end_year=2014 #2014
    # dictionaries for errors
    cols = range(start_year, end_year, 1)
    r2 = pd.DataFrame(columns=cols)
    mae = pd.DataFrame(columns=cols)
    mse = pd.DataFrame(columns=cols)

    # RefactorReesData()
    for year in range(2003, 2014, 1):
        print("*********************")
        print(year)
        dataset = DataSet(INPUT_PATH, train_year=year,  feature_sel=0)
        r2_dic, mae_dic, mse_dic = TestlLinearEachYear(dataset)
        r2 = r2.append(pd.Series(r2_dic, index=r2_dic.keys()), ignore_index=True)
        mae = mae.append(pd.Series(mae_dic, index=mae_dic.keys()), ignore_index=True)
        mse = mse.append(pd.Series(mse_dic, index=mse_dic.keys()), ignore_index=True)

    if not os.path.exists("./Results/"):
        os.makedirs("./Results/")

    r2.to_csv("./Results/"+station+"/Linear_r2.csv")
    mae.to_csv("./Results/"+station+"/Linear_mae.csv")
    mse.to_csv("./Results/"+station+"/Linear_mse.csv")

    """ calculate summary tables"""
    columns = ["Linear_r2", "Linear_mae", "Linear_mse"]
    summary_table = pd.DataFrame(columns=columns)

    for year_t in cols:
        temp = []
        index = year_t - cols[0]
        # xgboost_r2_mean = ((xgboost_r2.sum(axis=1))[index] - xgboost_r2.iloc[index, index]) / (len(cols) - 1)
        # xgboost_r2_mean = ((xgboost_r2.sum(axis=1))[index] - xgboost_r2.iloc[index, index]) / (len(cols) - 1)
        temp.append(((r2.sum(axis=1))[index] - r2.iloc[index, index]) / (len(cols) - 1))
        temp.append(((mae.sum(axis=1))[index] - mae.iloc[index, index]) / (len(cols) - 1))
        temp.append(((mse.sum(axis=1))[index] - mse.iloc[index, index]) / (len(cols) - 1))

        # summary_table=summary_table.append(np.reshape(temp,(1,-1)))
        summary_table = summary_table.append(pd.Series(temp, index=columns), ignore_index=True)
    summary_table.to_csv("./Results/"+station+"/Linear_summary.csv")

def TestlLinearEachYear(dataset):
    X_train, y_train,X_val,y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()

    modelFolder="./Models/"+station+"_models_"+str(dataset.train_year)+"/Linear/automl.pkl"
    model = pickle.load(open(modelFolder, 'rb'))

    # for i, value in enumerate(dataset.features):
    #     print(value,":",model.coef_[0,i])
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)

    r2_dic = {}
    mae_dic = {}
    mse_dic = {}
    for year in range(2003, 2014, 1):
        if year == dataset.train_year:
            x_data = X_train
            y_data = y_train
        else:
            x_data, y_data = dataset.GetEra5_Texas_TestData(year)

        y_pred = model.predict(x_data)
        # y_pred = model.predict(np.delete(x_data,22,axis=1))
        r2, mse, mae = dataset.EvaluateResults(modelFolder, x_data, y_data, y_pred)
        r2_dic[year] = r2
        mae_dic[year] = mae
        mse_dic[year] = mse
    return r2_dic, mae_dic, mse_dic

def TestlFeatureEachYear(dataset,feature):
    X_train, y_train,X_val,y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()

    r2_dic = {}
    mae_dic = {}
    mse_dic = {}
    for year in range(2003, 2014, 1):
        if year == dataset.train_year:
            x_data = X_train
            y_data = y_train
        else:
            x_data, y_data = dataset.GetEra5_Texas_TestData(year)

        feature_index=list(dataset.features).index(feature)
        y_pred = x_data[:,feature_index]
        r2, mse, mae = dataset.EvaluateResults(feature, x_data, y_data, y_pred)
        r2_dic[year] = r2
        mae_dic[year] = mae
        mse_dic[year] = mse
    return r2_dic, mae_dic, mse_dic

def TestBaseFeaturesYearly(feature_list):
    # cols = range(start_year, end_year, 1)

    for feature in feature_list:
        cols = [feature+"_r2",feature+"_mae",feature+"_mse"]
        dic = {}
        results_pd = pd.DataFrame(columns=cols)

        dataset = DataSet(INPUT_PATH, train_year=2003, feature_sel=0)
        X_train, y_train, X_val, y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()

        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)
        for year in range(2003, 2014, 1):
            print("*********************")
            print(feature +" "+ str(year))
            if year == dataset.train_year:
                x_data = X_train
                y_data = y_train
            else:
                x_data, y_data = dataset.GetEra5_Texas_TestData(year)

            feature_index = list(dataset.features).index(feature)
            y_pred = x_data[:, feature_index]

            r2_sc, mse_sc, mae_sc = dataset.EvaluateResults(feature, x_data, y_data, y_pred)
            dic[feature+"_r2"]=r2_sc
            dic[feature + "_mae"] = mae_sc
            dic[feature + "_mse"] = mse_sc
            results_pd = results_pd.append(pd.Series(dic), ignore_index=True)

        if not os.path.exists("./Results/"+ station):
            os.makedirs("./Results/"+ station)
        results_pd.to_csv("./Results/"+ station + "/"+feature+".csv")