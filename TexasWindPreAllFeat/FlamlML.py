from flaml import AutoML
import matplotlib.pyplot as plt
import pandas as pd
from TexasData import DataSet
import os
import pickle
from flaml.ml import sklearn_metric_loss_score
from Parameters import *
import xlsxwriter


def TrainFlamlML(dataset,estimator=['xgboost']):
    X_train, y_train,X_val,y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()

    modelFolder="./Models/"+station+"_models_"+str(dataset.train_year)+"/"
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)

    modelFolder = modelFolder + estimator[0]
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)

    automl = AutoML()

    settings = {
        "time_budget": time_budget,  # total running time in seconds
        "metric": 'mse',  # primary metrics for regression can be chosen from: ['mae','mse','r2','rmse','mape']
        "estimator_list":estimator,  # list of ML learners; we tune xgboost in this example
        "task": 'regression',  # task type
        "log_file_name": 'houses_experiment.log',  # flaml log file
        "seed":1
    }

    automl.fit( X_train=X_train, y_train=y_train, X_val=X_val,y_val=y_val, **settings)

    # pickle and save the automl object
    with open(modelFolder+'/automl.pkl', 'wb') as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    # compute predictions of testing dataset
    # y_pred = automl.predict(X_test)

    # compute different metric values on testing dataset
    # print('r2', '=', 1 - sklearn_metric_loss_score('r2', y_pred, y_test))
    # print('mse', '=', sklearn_metric_loss_score('mse', y_pred, y_test))
    # print('mae', '=', sklearn_metric_loss_score('mae', y_pred, y_test))


def TestFlamlML(dataset,estimator=['xgboost']):
    X_train, y_train,X_val,y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
    modelFolder="./Models/"+station+"_models_"+str(dataset.train_year)+"/"+estimator[0]+'/automl.pkl'

    automl=pickle.load(open(modelFolder, 'rb'))
    y_pred = automl.predict(X_test)

    modelFolder = "./models_"+str(dataset.train_year)+"/"+estimator[0]
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)

    r2,mse,mae=dataset.EvaluateResults(modelFolder, X_test, y_test, y_pred)
    f = open(modelFolder+  "_test_scores.txt", "w")
    f.write("r2="+ str(r2)+'\n')
    f.write("mse="+ str(mse)+'\n')
    f.write("mae="+ str(mae)+'\n')
    f.close()

def TestFlamlMLEachYear(dataset,estimator=['xgboost']):
    X_train, y_train,X_val,y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
    modelFolder="./Models/"+station+"_models_"+str(dataset.train_year)+"/"+estimator[0]
    automl=pickle.load(open(modelFolder+'/automl.pkl', 'rb'))

    r2_dic={}
    mae_dic={}
    mse_dic={}
    for year in range(start_year, end_year,1):
        print(year)
        if year==dataset.train_year:
            x_data=X_train
            y_data=y_train
        else:
            x_data,y_data=dataset.GetEra5_Texas_TestData(year)

        y_pred = automl.predict(x_data)
        r2,mse,mae=dataset.EvaluateResults(modelFolder, x_data, y_data, y_pred)
        r2_dic[year]=r2
        mae_dic[year]=mae
        mse_dic[year]=mse
    return r2_dic,mae_dic,mse_dic

def TestFlamMLYearly(INPUT_PATH,feature_sel):
    # dictionaries for errors
    cols=range(start_year, end_year, 1)
    xgboost_r2 = pd.DataFrame(columns=cols)
    xgboost_mae = pd.DataFrame(columns=cols)
    xgboost_mse = pd.DataFrame(columns=cols)

    rf_r2 = pd.DataFrame(columns=cols)
    rf_mae = pd.DataFrame(columns=cols)
    rf_mse = pd.DataFrame(columns=cols)

    lgbm_r2 = pd.DataFrame(columns=cols)
    lgbm_mae = pd.DataFrame(columns=cols)
    lgbm_mse = pd.DataFrame(columns=cols)

    extra_tree_r2 = pd.DataFrame(columns=cols)
    extra_tree_mae = pd.DataFrame(columns=cols)
    extra_tree_mse = pd.DataFrame(columns=cols)

    # RefactorReesData()
    for year in cols:
        print("*********************")
        print(year)
        dataset = DataSet(INPUT_PATH, train_year=year, feature_sel=feature_sel)

        r2_dic, mae_dic, mse_dic = TestFlamlMLEachYear(dataset, ["xgboost"])
        xgboost_r2 = xgboost_r2.append(pd.Series(r2_dic, index=r2_dic.keys()), ignore_index=True)
        xgboost_mae = xgboost_mae.append(pd.Series(mae_dic, index=mae_dic.keys()), ignore_index=True)
        xgboost_mse = xgboost_mse.append(pd.Series(mse_dic, index=mse_dic.keys()), ignore_index=True)

        r2_dic, mae_dic, mse_dic = TestFlamlMLEachYear(dataset, ["rf"])
        rf_r2 = rf_r2.append(pd.Series(r2_dic, index=r2_dic.keys()), ignore_index=True)
        rf_mae = rf_mae.append(pd.Series(mae_dic, index=mae_dic.keys()), ignore_index=True)
        rf_mse = rf_mse.append(pd.Series(mse_dic, index=mse_dic.keys()), ignore_index=True)

        r2_dic, mae_dic, mse_dic = TestFlamlMLEachYear(dataset, ["lgbm"])
        lgbm_r2 = lgbm_r2.append(pd.Series(r2_dic, index=r2_dic.keys()), ignore_index=True)
        lgbm_mae = lgbm_mae.append(pd.Series(mae_dic, index=mae_dic.keys()), ignore_index=True)
        lgbm_mse = lgbm_mse.append(pd.Series(mse_dic, index=mse_dic.keys()), ignore_index=True)

        r2_dic, mae_dic, mse_dic = TestFlamlMLEachYear(dataset, ["extra_tree"])
        extra_tree_r2 = extra_tree_r2.append(pd.Series(r2_dic, index=r2_dic.keys()), ignore_index=True)
        extra_tree_mae = extra_tree_mae.append(pd.Series(mae_dic, index=mae_dic.keys()), ignore_index=True)
        extra_tree_mse = extra_tree_mse.append(pd.Series(mse_dic, index=mse_dic.keys()), ignore_index=True)

    if not os.path.exists("./Results/"+station+"/"):
        os.makedirs("./Results/"+station+"/")

    xgboost_r2.to_csv("./Results/"+station+"/xgboost_r2.csv")
    xgboost_mae.to_csv("./Results/"+station+"/xgboost_mae.csv")
    xgboost_mse.to_csv("./Results/"+station+"/xgboost_mse.csv")

    rf_r2.to_csv("./Results/"+station+"/rf_r2.csv")
    rf_mae.to_csv("./Results/"+station+"/rf_mae.csv")
    rf_mse.to_csv("./Results/"+station+"/rf_mse.csv")

    lgbm_r2.to_csv("./Results/"+station+"/lgbm_r2.csv")
    lgbm_mae.to_csv("./Results/"+station+"/lgbm_mae.csv")
    lgbm_mse.to_csv("./Results/"+station+"/lgbm_mse.csv")

    extra_tree_r2.to_csv("./Results/"+station+"/extra_tree_r2.csv")
    extra_tree_mae.to_csv("./Results/"+station+"/extra_tree_mae.csv")
    extra_tree_mse.to_csv("./Results/"+station+"/extra_tree_mse.csv")

    """ calculate summary tables"""
    # columns = ["xgboost_r2", "rf_r2", "lgbm_r2", "xgboost_mae", "rf_mae", "lgbm_mae", "xgboost_mse", "rf_mse","lgbm_mse"]
    # columns = ["rf_r2", "xgboost_r2", "lgbm_r2","extra_tree_r2","catboost_r2","rf_mae",  "xgboost_mae", "lgbm_mae","extra_tree_mae","catboost_mae", "rf_mse","xgboost_mse", "lgbm_mse","extra_tree_mse","catboost_mse"]
    columns = ["rf_r2", "xgboost_r2", "lgbm_r2","extra_tree_r2","rf_mae",  "xgboost_mae", "lgbm_mae","extra_tree_mae", "rf_mse","xgboost_mse", "lgbm_mse","extra_tree_mse"]
    summary_table = pd.DataFrame(columns=columns)

    for year_t in cols:
        temp=[]
        index=year_t-cols[0]
        # xgboost_r2_mean = ((xgboost_r2.sum(axis=1))[index] - xgboost_r2.iloc[index, index]) / (len(cols) - 1)
        # xgboost_r2_mean = ((xgboost_r2.sum(axis=1))[index] - xgboost_r2.iloc[index, index]) / (len(cols) - 1)
        temp.append(((rf_r2.sum(axis=1))[index]-rf_r2.iloc[index,index])/(len(cols)-1))
        temp.append(((xgboost_r2.sum(axis=1))[index]-xgboost_r2.iloc[index,index])/(len(cols)-1))
        temp.append(((lgbm_r2.sum(axis=1))[index]-lgbm_r2.iloc[index,index])/(len(cols)-1))
        temp.append(((extra_tree_r2.sum(axis=1))[index]-extra_tree_r2.iloc[index,index])/(len(cols)-1))
        # temp.append(((catboost_r2.sum(axis=1))[index]-catboost_r2.iloc[index,index])/(len(cols)-1))

        temp.append(((rf_mae.sum(axis=1))[index]-rf_mae.iloc[index,index])/(len(cols)-1))
        temp.append(((xgboost_mae.sum(axis=1))[index]-xgboost_mae.iloc[index,index])/(len(cols)-1))
        temp.append(((lgbm_mae.sum(axis=1))[index]-lgbm_mae.iloc[index,index])/(len(cols)-1))
        temp.append(((extra_tree_mae.sum(axis=1))[index]-extra_tree_mae.iloc[index,index])/(len(cols)-1))
        # temp.append(((catboost_mae.sum(axis=1))[index]-catboost_mae.iloc[index,index])/(len(cols)-1))

        temp.append(((rf_mse.sum(axis=1))[index]-rf_mse.iloc[index,index])/(len(cols)-1))
        temp.append(((xgboost_mse.sum(axis=1))[index]-xgboost_mse.iloc[index,index])/(len(cols)-1))
        temp.append(((lgbm_mse.sum(axis=1))[index]-lgbm_mse.iloc[index,index])/(len(cols)-1))
        temp.append(((extra_tree_mse.sum(axis=1))[index]-extra_tree_mse.iloc[index,index])/(len(cols)-1))
        # temp.append(((catboost_mse.sum(axis=1))[index]-catboost_mse.iloc[index,index])/(len(cols)-1))

        # summary_table=summary_table.append(np.reshape(temp,(1,-1)))
        summary_table=summary_table.append(pd.Series(temp,index=columns), ignore_index=True)
    summary_table.to_csv("./Results/"+station+"/FLAML_summary.csv")

def PrintBestConfig():
    for model in models: #,'xgboost'
        for year in range(start_year, end_year, 1):
            modelFolder="./Models/"+station+"_models_"+str(year) + "/" + model + '/automl.pkl'
            automl = pickle.load(open(modelFolder, 'rb'))
            print("********* "+model+" __ "+str(year) +"***********")
            print(automl.best_config)

            print("-------------------------------")

def prepareWorksheet(worksheet,row,col,scat_name,model):
    worksheet.write(row, col, 'Dataset')
    worksheet.write(row+2, col, scat_name)
    col = col + 1
    worksheet.write(row, col, 'Model')
    col = col + 1
    worksheet.write(row, col, 'Years')
    row=row+1

    col = col-1
    col = col+1
    for year in range(start_year, end_year):
        worksheet.write(row, col, year)
        col = col + 1
    worksheet.write(row, col, "mean")

    return  col

def writetoWorksheet(worksheet,scat_name,model,model_path,row=0,col=0,new=0):
    inc=1
    if new==1:
        prepareWorksheet(worksheet, row, col, scat_name, model)
        inc=2
    # R2

    INPUT_PATH = "./Results/" + scat_name + model_path
    data = pd.read_csv(INPUT_PATH, index_col=0)
    col = 2
    mean = 0
    worksheet.write(row+inc, col-1, model)
    for i in range(11):
        worksheet.write(row+inc, col,  round(data[model+"_r2"][i],2))
        mean = mean + data[model+"_r2"][i]
        col = col + 1
    worksheet.write(row+inc, col,  round(mean / (i + 1),2))

    # MAE
    col = col + 2
    if new==1:
        prepareWorksheet(worksheet, row, col, scat_name, model)
    col = 17
    mean = 0
    worksheet.write(row+inc, col-1, model)
    for i in range(11):
        worksheet.write(row+inc, col,  round(data[model+"_mae"][i],2))
        mean = mean + data[model+"_mae"][i]
        col = col + 1
    worksheet.write(row+inc, col,  round(mean / (i + 1),2))

    # MSE
    col = col + 2
    if new==1:
        prepareWorksheet(worksheet, row, col, scat_name, model)
    col = 32
    mean = 0
    worksheet.write(row+inc, col-1, model)
    for i in range(11):
        worksheet.write(row+inc, col,  round(data[model+"_mse"][i],2))
        mean = mean + data[model+"_mse"][i]
        col = col + 1
    worksheet.write(row+inc, col, round(mean / (i + 1),2))
