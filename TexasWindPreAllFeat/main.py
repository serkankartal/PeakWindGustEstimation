from FlamlML import *
from RefactorDataFiles import *
from ML import *
from FeatureImportanceML import *
from Eli5Test import *
torch.cuda.is_available()
from TexasData import DataSet
from VisualizeModels import *
from Parameters import *

INPUT_PATH = "./DATA/ERA5_" + station + "_merged_time_coded.csv"

def CalculateFeatureImportance():
    for year in range(start_year,end_year,1):
        print("*********************")
        print(station+" "+str(year))
        dataset=DataSet(INPUT_PATH,train_year=year,feature_sel=appy_feature_sel)

        for est in models:#,'xgboost', "lgbm" "rf","lgbm",'extra_tree','catboost'
            print("ELI5 is calculting for "+station + "_models_" + str(year)+'/'+est)
            CalEli5Features(dataset,"./Models/" + station + "_models_" + str(year)+'/'+est+'/')
            if year==(end_year-1):
                print("ELI5 average score is calculting for "+station )
                CalAverageEli5(est)


def TrainTestModels():
    for year in range(start_year, end_year, 1):
        print("*********************")
        print(station + " " + str(year))
        dataset = DataSet(INPUT_PATH, train_year=year, feature_sel=appy_feature_sel)
        # VisualizeFlamlML(dataset)

        """Train and test models"""
        for est in models:  # ,'catboost' "lgbm",'extra_tree'
            TrainFlamlML(dataset, list(est.split(" ")))
        for est in models:  # , "rf", "lgbm"
            TestFlamlML(dataset, list(est.split(" ")))



if __name__ == '__main__':
    # RefactorReesData()"

    if appy_feature_sel:
        CalculateFeatureImportance()
    TrainTestModels()

