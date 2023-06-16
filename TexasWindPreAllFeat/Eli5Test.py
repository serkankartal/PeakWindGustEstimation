import matplotlib.pyplot as plt
import pickle
import numpy as np
from flaml.ml import sklearn_metric_loss_score
from eli5.permutation_importance import get_score_importances
import os
from Parameters import *


def score(X, y):
    global full_model_name
    automl = pickle.load(open(full_model_name, 'rb'))
    y_pred = automl.predict(X)
    r2 = 1 - sklearn_metric_loss_score('r2', y_pred, y)
    return r2

def CalEli5Features(dataset, model_path,model_name):
    X_train, y_train,X_val,y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
    global full_model_name
    full_model_name=model_path

    if not os.path.exists("./eli5/eli5_scores_"+station):
        os.mkdir("./eli5/eli5_scores_"+station)

    base_score, score_decreases = get_score_importances(score, X_test, y_test)
    feature_importances = np.mean(score_decreases, axis=0)
    print(feature_importances)
    np.savetxt("./eli5/eli5_scores_"+station+"/"+model_name+"_"+ str(dataset.train_year) +"_eli_feature_scores.txt", feature_importances, fmt='%.3f', delimiter=',')

def CalAverageEli5(model_name):

    file_list=os.listdir("./eli5/eli5_scores_"+station)
    eli_scores=None
    for file in file_list:
        score_temp=np.loadtxt("./eli5/eli5_scores_"+station+"/"+file,delimiter=",")
        if eli_scores is None:
            eli_scores=score_temp
        else:
            eli_scores=np.append(eli_scores.reshape(-1,265),score_temp.reshape(1,-1),axis=0)

    mean=eli_scores.mean(axis=0)
    np.savetxt("./eli_feature_scores.txt", mean, delimiter=",", fmt = '%.3f')
    # mean.to_csv(index=False)

