start_year=2003
end_year=2004 #2014
station="Rees" #"Rees", "Macy","Fluvanna"
models=["xgboost"]#["xgboost", "rf", "lgbm"]
appy_feature_sel=0
feature_count=10
weighted_features=True
threshold=20
time_budget= 3600  #FLAML total running time in seconds default 3600
outlier_fraction_mult=5 # 2 means number of outlierin training data will be multiplied by 2 and applied to the test data
