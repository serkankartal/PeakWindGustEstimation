import matplotlib.pyplot as plt
import pickle
import pandas as pd

def CalFeatureImportance(dataset,model_path):
    X_train, y_train,X_val,y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
    automl = pickle.load(open(model_path+"automl.pkl", 'rb'))
    # plt.barh(dataset.features, automl.model.estimator.feature_importances_)

    feature_imp = pd.DataFrame(columns=dataset.features)

    feature_imp = feature_imp.append(pd.Series(automl.model.estimator.feature_importances_, index=dataset.features), ignore_index=True)
    feature_imp.to_csv(model_path+"feature_scores.txt")

