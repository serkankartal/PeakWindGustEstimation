from dtreeviz.trees import *
# from IPython.display import Image, display_svg, SVG
import pickle


def VisualizeFlamlML(dataset):
    X_train, y_train,X_val,y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
    for est in ["rf", "lgbm"]:  # ,'xgboost''xgboost',
        modelFolder = "./models_" + str(dataset.train_year) + "/" + est+ '/automl.pkl'
        regr = pickle.load(open(modelFolder, 'rb'))
        viz = dtreeviz(regr.model,
                       X_train,
                       y_train,
                       target_name='Wind',  # this name will be displayed at the leaf node
                       feature_names=dataset.features,
                       title="ERA5 - Texas_Rees",
                       fontname="Arial",
                       title_fontsize=16,
                       colors={"title": "purple"}
                       )
        s=4