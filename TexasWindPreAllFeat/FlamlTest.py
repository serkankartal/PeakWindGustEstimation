from flaml import AutoML
# from flaml.default import
from TabNetTexas import *

# INPUT_PATH = "./DATA/ERA5_Rees_merged_time.csv"
INPUT_PATH="./DATA/ERA5_Rees_merged_daily_mean.csv"

dataset = DataSet(INPUT_PATH)
X_train, y_train, X_valid, y_valid, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
X_train = np.concatenate((X_train, X_valid))
y_train = np.concatenate((y_train, y_valid))

automl = AutoML()
settings = {
    "time_budget": 120,  # total running time in seconds
    "metric": 'r2',  # primary metrics for regression can be chosen from: ['mae','mse','r2','rmse','mape']
    "estimator_list": ['xgboost'],  # list of ML learners; we tune xgboost in this example
    "task": 'regression',  # task type
    "log_file_name": 'houses_experiment.log',  # flaml log file
}

automl.fit(X_train=X_train, y_train=y_train, **settings)
# retrieve best config
print('Best hyperparmeter config:', automl.best_config)
print('Best r2 on validation data: {0:.4g}'.format(1 - automl.best_loss))
print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))

# plot feature importance
# import matplotlib.pyplot as plt
# plt.barh(X_train.columns, automl.model.estimator.feature_importances_)
# automl.fit(X_train, y_train, task="regression", estimator_list=["lgbm"])

# compute predictions of testing dataset
y_pred = automl.predict(X_test)
# print('Predicted labels', y_pred)
# print('True labels', y_test)
y_pred = dataset.UnScaleData(X_test[:, 1:], np.reshape(y_pred, (y_pred.shape[0], 1)))
y_test = dataset.UnScaleData(X_test[:, 1:], np.reshape(y_test, (y_test.shape[0], 1)))
# compute different metric values on testing dataset
from flaml.ml import sklearn_metric_loss_score
print('r2', '=', 1 - sklearn_metric_loss_score('r2', y_pred, y_test))
print('mse', '=', sklearn_metric_loss_score('mse', y_pred, y_test))
print('mae', '=', sklearn_metric_loss_score('mae', y_pred, y_test))
