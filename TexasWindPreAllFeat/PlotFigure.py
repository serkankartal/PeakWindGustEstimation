import matplotlib.pyplot as plt

from RefactorDataFiles import *
import seaborn as sn; sn.set()# rc={'text.usetex' : True}
from TexasData import DataSet
from statsmodels.distributions.empirical_distribution import ECDF
import pickle
from scipy.stats import pearsonr,probplot
from TraditionalMethods import *
from matplotlib import pyplot as plt
from numpy import hstack
from cycler import cycler
from VisualizeModels import *
from statsmodels.graphics.gofplots import qqplot_2samples

import statsmodels.api as sm
import pylab as py

import matplotlib as mpl

def plotFigure2():
    if not os.path.exists("./Figures/peakGustPlots/"):
        os.makedirs("./Figures/peakGustPlots/")

    drawTimeRangefor5minData(year=2006, month=5, day=20, hour=15, hour_margin=2)
    drawTimeRangefor5minData(year=2008, month=5, day=21, hour=21, hour_margin=2)
    drawTimeRangefor5minData(year=2008, month=6, day=19, hour=22, hour_margin=2)
    drawTimeRangefor5minData(year=2008, month=8, day=14, hour=20, hour_margin=2)
    drawTimeRangefor5minData(year=2009, month=6, day=4, hour=21, hour_margin=2)
    drawTimeRangefor5minData(year=2009, month=8, day=12, hour=18, hour_margin=2)

def plotFigure3():
    location_name=["REESE","MACY","FLUVANNA"]
    if not os.path.exists("./Figures/Histogram/"):
        os.makedirs("./Figures/Histogram/")

    for i,scat_name in enumerate(["Rees","Macy","Fluvanna"]):
        INPUT_PATH="./DATA/ERA5_"+scat_name+"_merged_time_coded.csv"
        dataset=DataSet(INPUT_PATH,train_year=2003,feature_sel=0)
        np.random.seed(1)

        """Load data and split"""
        train = pd.read_csv(dataset.DataFilePath)
        train = train.drop(train[train[dataset.target] < 0].index.tolist(), axis=0)

        train=train.reset_index()
        train=train.drop("index",axis=1)
        if  "TIME" in train:
            train=train.drop("TIME",axis=1)


        y_label="${W_p \: (ms}^{-1})$ - "+location_name[i]
        indices = train[(train.Year> 2) & (train.Year<15)].index
        fig, ax = plt.subplots()

        plt.figure(figsize=(14,13))

        ax.yaxis.set_tick_params(labelsize=20)
        ax.xaxis.set_tick_params(labelsize=10)
        sn.set(style="white",font_scale=1.5)
        # sn.set(style="whitegrid", font_scale=1.5)
        # rc = {'axes.labelsize': 16, 'font.size': 14, 'legend.fontsize': 14.0, 'axes.titlesize': 14,'style':"whitegrid"}

        era5_feature="i10fg" #ERA5_GUST_10m
        # era5_label="$\mathregular{u_*}(m s^{-1})$"
        # era5_label="$\mathregular{H} (m)$"
        era5_label="${W_{p10}^{i}\:(ms}^{-1})$ - ERA5"
        # era5_label="$\mathregular{W_{i10}}\: (ms^{-1})$ - ERA5"
        g = sn.jointplot(y=train["max"][indices], x=train[era5_feature][indices],marginal_ticks=True,  marginal_kws = dict(bins=50),
                         kind="hex", color="darkorange", xlim=(0,25), ylim=(0,25), space=0,)
        # g.set_axis_labels(era5_label,"wind gust")
        plt.xlabel(era5_label)
        plt.ylabel(y_label)
        plt.yticks([0,5,10,15,20])
        plt.xticks([0,5,10,15,20])
        pr2=pearsonr(train["max"][indices],train[era5_feature][indices])
        plt.text(19,1,"${r}^{2}$=%0.2f"%pr2[0])
        g.savefig("./Figures/Histogram/"+scat_name+"_"+era5_feature+"_histogram_plot.png", dpi=300,bbox_inches='tight')
        g.savefig("./Figures/Histogram/"+scat_name+"_"+era5_feature+"_histogram_plot.eps", format='eps',bbox_inches='tight')

        # second figure
        era5_feature="zust" #ERA5_UST
        era5_label="${u_*\: (m s}^{-1})$ - ERA5"
        # era5_label="$\mathregular{H} (m)$"
        g = sn.jointplot(y=train["max"][indices], x=train[era5_feature][indices],marginal_ticks=True,  marginal_kws = dict(bins=50),
                         kind="hex", color="darkorange", xlim=(0,1), ylim=(0,25), space=0,)
        # g.set_axis_labels(era5_label,"wind gust")
        plt.xlabel(era5_label)
        plt.ylabel(y_label)
        plt.yticks([0,5,10,15,20])
        plt.xticks([0,0.2,0.4,0.6,0.8])
        pr2=pearsonr(train["max"][indices],train[era5_feature][indices])
        plt.text(0.76,1,"${r}^{2}$=%0.2f"%pr2[0])
        g.savefig("./Figures/Histogram/"+scat_name+"_"+era5_feature+"_histogram_plot.png", dpi=300,bbox_inches='tight')
        g.savefig("./Figures/Histogram/"+scat_name+"_"+era5_feature+"_histogram_plot.eps", format='eps',bbox_inches='tight')

        # uthird figure
        era5_feature="blh" # ERA5_PBLH
        era5_label="${H\:(m)}$- ERA5"
        g = sn.jointplot(y=train["max"][indices], x=train[era5_feature][indices],marginal_ticks=True,  marginal_kws = dict(bins=50),
                         kind="hex", color="darkorange", xlim=(0,3000), ylim=(0,25), space=0,)
        # g.set_axis_labels(era5_label,"wind gust")
        plt.xlabel(era5_label)
        plt.ylabel(y_label)
        plt.yticks([0,5,10,15,20])
        plt.xticks([0,1000,2000])
        pr2=pearsonr(train["max"][indices],train[era5_feature][indices])
        plt.text(2280,1,"${r}^{2}$=%0.2f"%pr2[0])
        g.savefig("./Figures/Histogram/"+scat_name+"_"+era5_feature+"_histogram_plot.png", dpi=300,bbox_inches='tight')
        g.savefig("./Figures/Histogram/"+scat_name+"_"+era5_feature+"_histogram_plot.eps", format='eps',bbox_inches='tight')

def plotqq():
    location_name=["REESE","MACY","FLUVANNA"]
    if not os.path.exists("./Figures/qqplots/"):
        os.makedirs("./Figures/qqplots/")
    model_name = ["XGB", "RF", "LGBM"]
    for year in range(2003,2014,1):
        for i,scat_name in enumerate(["Rees","Macy","Fluvanna"]):
            INPUT_PATH="./DATA/ERA5_"+scat_name+"_merged_time_coded.csv"
            dataset=DataSet(INPUT_PATH,train_year=year,feature_sel=0)


            X_train, y_train, X_val, y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
            y = dataset.UnScaleData(X_test,y_test)
            pp_x = sm.ProbPlot(y)

            ERA5_GUST_10m=dataset.UnScaleData(X_test,X_test[:,2])
            pp_era=sm.ProbPlot(ERA5_GUST_10m) # ERA5_GUST_10m 2. indexe denk gelmektedir

            sn.set(style="white", font_scale=1, palette=["black"])
            for im,est in enumerate(['xgboost',"rf", "lgbm"]): #, "rf", "lgbm"
                modelFolder = "./models_" + str(dataset.train_year) + "/" + est + '/automl.pkl'

                automl = pickle.load(open(modelFolder, 'rb'))
                y_pred = automl.predict(X_test)
                y_pred = dataset.UnScaleData(X_test, y_pred)
                pp_y = sm.ProbPlot(y_pred)

                fig = plt.figure()
                # ax = fig.add_axes([0, 0])
                ax = fig.add_subplot()
                qqplot_2samples(pp_x, pp_y, xlabel=None, ylabel=None, line=None, ax=ax)
                # mpl.rcParams['axes.prop_cycle'] = cycler(color=['k','r', 'g',  'y'])
                qqplot_2samples(pp_x, pp_era, xlabel=None, ylabel=None, line=None, ax=ax)

                ax.get_lines()[0].set_markerfacecolor('k')
                ax.get_lines()[0].set_markeredgecolor('k')
                ax.get_lines()[0].set_markersize(4.0)

                ax.get_lines()[1].set_markerfacecolor('r')
                ax.get_lines()[1].set_markeredgecolor('r')
                ax.get_lines()[1].set_markersize(4.0)

                ax.legend(ax.get_lines(), ['Prediction', 'ERA5'])
                py.ylabel(model_name[im]+" prediction")
                py.xlabel(location_name[i]+" - wind gust")

                py.xlim(0,50)
                py.ylim(0,50)

                # py.show()
                py.savefig("./Figures/qqplots/"+scat_name+"_"+model_name[im]+"_"+str(year)+"_qq_plot.png", dpi=300,bbox_inches='tight')
                py.savefig("./Figures/qqplots/"+scat_name+"_"+model_name[im]+"_"+str(year)+"_qq_plot.eps", format='eps',bbox_inches='tight')

def plotECDF():
    if not os.path.exists("./Figures/ECDFplots/"):
        os.makedirs("./Figures/ECDFplots/")
    model_name = ["XGB", "RF", "LGBM"]
    for scat_name in ["Rees"]:#,"Macy","Fluvanna"
        INPUT_PATH="./DATA/ERA5_"+scat_name+"_merged_time_coded.csv"
        dataset=DataSet(INPUT_PATH,train_year=2003,feature_sel=0)

        X_train, y_train, X_val, y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
        y = dataset.UnScaleData(X_test,y_test)
        sn.set(style="white", font_scale=1.5, palette=["black"])

        for im,est in enumerate(['xgboost',"rf", "lgbm"]): #, "rf", "lgbm"
            modelFolder = "./models_" + str(dataset.train_year) + "/" + est + '/automl.pkl'

            automl = pickle.load(open(modelFolder, 'rb'))
            y_pred = automl.predict(X_test)
            y_pred = dataset.UnScaleData(X_test, y_pred)

            # fit a cdf
            ecdf = ECDF(y_pred)
            plt.plot(ecdf.x, ecdf.y,color="red")

            ecdf = ECDF(y)
            plt.plot(ecdf.x, ecdf.y,color="blue")

            py.xlabel("wind gust vs "+est+" prediction")
            py.xlim(0,25)
            py.ylim(0,1)

            # py.show()
            py.savefig("./Figures/ECDFplots/"+scat_name+"_"+model_name[im]+"_qq_plot.png", dpi=300,bbox_inches='tight')
            py.savefig("./Figures/ECDFplots/"+scat_name+"_"+model_name[im]+"_qq_plot.eps", format='eps',bbox_inches='tight')
            plt.cla()

def PlotEli5Features():
    location_name=["MACY","REESE","FLUVANNA"]
    model_name=["XGB","RF"]
    if not os.path.exists("./Figures/Eli5/"):
        os.makedirs("./Figures/Eli5/")

    listofFeatures=[]
    for i,scat_name in enumerate(["Rees","Macy","Fluvanna"]): #
        INPUT_PATH="./DATA/ERA5_"+scat_name+"_merged_time_coded.csv"
        dataset=DataSet(INPUT_PATH,train_year=2003,feature_sel=0)
        X_train, y_train, X_val, y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
        file_list=os.listdir("./Eli5/eli5_scores_"+scat_name)
        eli_scores_mean=None
        # cols=np.delete(dataset.columns.to_numpy(), 22)

        # cols=["WSPD 10m","WSPD 100m","Gust 10m","$\mathregular{\\alpha}$","Beta","T 2m",
        #         "TSK","TSL","Td 2m","dT1","dT2","dT3","UST","SHFX","LH","PMSL",
        #         "PBLH","TCC","LCC","EDR","CAPE","CIN","Hour Sin","Hour Cos",
        #         "Day Sin","Day Cos","Month Sin","Month Cos"]

        # cols=[   "UST","SHFX","LH","PMSL",
        #         "PBLH","TCC","LCC","EDR","CAPE","CIN","Hour Sin","Hour Cos",
        #         "Day Sin","Day Cos","Month Sin","Month Cos"]
        #
        # cols=[
        #       "$\mathregular{W}_{10}$",
        #       "$\mathregular{W}_{100}$",
        #       "$\mathregular{W}_{i10}$",
        #       "$\mathregular{\\alpha}$",
        #       "$\mathregular{\\beta}$",
        #       "$\mathregular{T}_{2}$",
        #       "$\mathregular{T}_{0}$",
        #       "$\mathregular{T}_{s}$",
        #       "$\mathregular{T}_{d2}$",
        #       "$\mathregular{\\Delta}\: {T}_{1}$",
        #       "$\mathregular{\\Delta}\: {T}_{2}$",
        #       "$\mathregular{\\Delta}\: {T}_{3}$",
        #       "$\mathregular{u}_{*}$",
        #
        #
        #       "$\mathregular{H}_{S}$",
        #       "$\mathregular{H}_{L}$",
        #       "$\mathregular{P}_{0}$",
        #       "$\mathregular{H}$",
        #       "TCC","LCC",
        #       "$\mathregular{\\overline{\\varepsilon}}$",
        #
        #       "CAPE","CIN",
        #       "HRSin","HRCos", "DYSin","DYCos", "MOSin","MOCos",
        #
        #       ]
        cols=dataset.features
        cols= [s.replace('ERA5_', '') for s in cols]
        for im,model in enumerate(['xgboost',"rf"]):
            eli_scores=None
            for file in file_list:
                if file.__contains__(model):
                    score_temp=np.loadtxt("./Eli5/eli5_scores_"+scat_name+"/"+file,delimiter=",")
                    if eli_scores is None:
                        eli_scores=score_temp
                    else:
                        eli_scores=np.append(eli_scores.reshape(-1,265),score_temp.reshape(1,-1),axis=0)
                        # eli_scores=np.append(eli_scores.reshape(-1,31),score_temp.reshape(1,-1),axis=0)

            mean=eli_scores.mean(axis=0)
            # mean= np.delete(mean, [23,24,25]).astype(float)
            temp=np.concatenate((np.asarray(cols).reshape(len(cols), -1),np.asarray(([model_name[im]]*len(cols))).reshape(-1,1), np.asarray(mean).reshape(-1,1)), axis=1)
            if eli_scores_mean is None:
                eli_scores_mean = temp
            else:
                eli_scores_mean = np.append(eli_scores_mean, temp, axis=0)

        num_of_feature=10
        pd_scores= pd.DataFrame(eli_scores_mean, columns=["features", "ML Models", "eli5_scores"])
        pd_scores["eli5_scores"] = pd_scores["eli5_scores"].astype(float)
        pd_scores= pd_scores.sort_values(by=['eli5_scores'], ascending=False)
        if len(listofFeatures)==0:
            pd_rf=pd_scores[pd_scores["ML Models"]=="RF"].sort_values(by=['eli5_scores'], ascending=False).iloc[:num_of_feature]
            listofFeatures=list(pd_scores[pd_scores["ML Models"]=="RF"].sort_values(by=['eli5_scores'], ascending=False).iloc[:num_of_feature,0])
            row_list=pd_scores[pd_scores["ML Models"]=="RF"]["features"].isin(listofFeatures)
            row_list=pd.Series(row_list)
        else:
            row_list=pd_scores[pd_scores["ML Models"]=="RF"]["features"].isin(listofFeatures)
            row_list=pd.Series(row_list)
            pd_rf=pd_scores[pd_scores["ML Models"]=="RF"]
            pd_rf=pd_rf[row_list.values]

        temp=pd_scores[pd_scores["ML Models"]=="RF"]
        others_sum=temp[row_list.values==False]["eli5_scores"].sum()
        others_rf = pd.DataFrame({"features": ["Others("+str(len(cols)-10)+")"], "ML Models": ["RF"], "eli5_scores": [others_sum]})
        pd_rf=pd.concat([pd_rf, others_rf])



        # others_sum=pd_scores[pd_scores["ML Models"]=="RF"].sort_values(by=['eli5_scores'], ascending=False).iloc[num_of_feature:]["eli5_scores"].sum()
        row_list = pd_scores[pd_scores["ML Models"] == "XGB"]["features"].isin(listofFeatures)
        row_list = pd.Series(row_list)
        pd_xgboost = pd_scores[pd_scores["ML Models"] == "XGB"]
        pd_xgboost = pd_xgboost[row_list.values]

        temp=pd_scores[pd_scores["ML Models"]=="XGB"]
        others_sum_xgb=temp[row_list.values==False]["eli5_scores"].sum()
        # others_sum_xgb=pd_scores[pd_scores["ML Models"]=="XGB"].sort_values(by=['eli5_scores'], ascending=False).iloc[num_of_feature:]["eli5_scores"].sum()
        others_xgb = pd.DataFrame({"features": ["Others("+str(len(cols)-10)+")"], "ML Models": ["XGB"], "eli5_scores": [others_sum_xgb]})
        pd_xgboost=pd.concat([pd_xgboost, others_xgb])

        # pd_xgboost = pd_scores[pd_scores["ML Models"] == "XGB"].sort_values(by=['eli5_scores'], ascending=False).iloc[:20]
        # pd_rf = pd_scores[pd_scores["ML Models"] == "RF"].sort_values(by=['eli5_scores'], ascending=False)
        # pd_rf = pd_rf[pd_rf['features'].isin(list(pd_xgboost["features"]))]

        if len(listofFeatures)==num_of_feature:
            listofFeatures.append("Others("+str(len(cols)-10)+")")
        pd_rf['findex'] = pd_rf.loc[:, 'features']
        pd_rf=pd_rf.set_index('findex')
        pd_rf = pd_rf.reindex(listofFeatures)

        pd_xgboost['findex'] = pd_xgboost.loc[:, 'features']
        pd_xgboost=pd_xgboost.set_index('findex')
        pd_xgboost = pd_xgboost.reindex(listofFeatures)

        pd_scores=pd.concat([pd_rf, pd_xgboost], axis=0)

        pd_scores=pd_scores.to_numpy()
        pd_scores[pd_scores[:,0]=='v100',0]="$\mathregular{V}_{100}$"
        pd_scores[pd_scores[:,0]=='inss',0]="$\\tau_{ns}$"
        pd_scores[pd_scores[:,0]=='zust',0]="$\mathregular{u}_{*}$"
        pd_scores[pd_scores[:,0]=='bld',0]= "$\mathregular{\\overline{\\varepsilon}}$"
        pd_scores[pd_scores[:,0]=='mbld',0]= "$\mathregular{\\overline{\\varepsilon}_m}$"
        pd_scores[pd_scores[:,0]=='WSPD_10m',0]="$\mathregular{W}_{10}$"
        pd_scores[pd_scores[:,0]=='WSPD_100m',0]="$\mathregular{W}_{100}$"
        pd_scores[pd_scores[:,0]=='i10fg',0]="$\mathregular{W}_{p10}^i$"
        pd_scores[pd_scores[:,0]=='fg10',0]="$\mathregular{W}_{p10}^m$"
        pd_scores[pd_scores[:,0]=='alpha',0]="$\mathregular{\\alpha}$"
        pd_scores[pd_scores[:,0]=='beta',0]="$\mathregular{\\beta}$"
        pd_scores[pd_scores[:,0]=='t2',0]="$\mathregular{T}_{2}$"
        pd_scores[pd_scores[:,0]=='skt',0]="$\mathregular{T}_{0}$"
        pd_scores[pd_scores[:,0]=='stl1',0]="$\mathregular{T}_{s}$"
        pd_scores[pd_scores[:,0]=='ishf',0]= "$\mathregular{H}_{S}$"
        pd_scores[pd_scores[:,0]=='ie',0]="$\mathregular{H}_{L}$"
        pd_scores[pd_scores[:,0]=='msl',0]="$\mathregular{P}_{0}$"
        pd_scores[pd_scores[:,0]=='blh',0]="$\mathregular{H}$"
        pd_scores[pd_scores[:,0]=='cape',0]= "CAPE"
        pd_scores[pd_scores[:,0]=='cin',0]="CIN"
        pd_scores[pd_scores[:,0]=='hour_sin',0]="HRSin"
        pd_scores[pd_scores[:,0]=='hour_cos',0]="HRCos"
        pd_scores[pd_scores[:,0]=='day_sin',0]="DYSin"
        pd_scores[pd_scores[:,0]=='day_cos',0]="DYCos"
        pd_scores[pd_scores[:,0]=='month_sin',0]="MOSin"
        pd_scores[pd_scores[:,0]=='month_cos',0]="MOCos"

        pd_scores = pd.DataFrame(pd_scores, columns=["features", "ML Models", "eli5_scores"])
        pd_scores["eli5_scores"] = pd_scores["eli5_scores"].astype(float)

        # sns.set(font_scale=0.2)
        sns.set_theme(style="whitegrid", font_scale=0.6)
        # sns.set_palette("Paired")
        # sns.color_palette("colorblind")

        # Draw a nested barplot by species and sex
        g = sns.catplot(data=pd_scores, kind="bar", palette=["darkorange","limegreen","blue"], #legend=False,
            x="eli5_scores", y="features", hue="ML Models",height=3 ,  aspect=8/12 )#alpha=.6,
        sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 0.97), ncol=2, title=None, frameon=False,)


        # g = sns.catplot(data=pd_scores, kind="bar", palette=["darkorange","limegreen","blue"], #legend=False,
        #     x="eli5_scores", y="features", hue="ML Models",height=6,  aspect=8/12  )#alpha=.6,
        # sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 0.97), ncol=2, title=None, frameon=False,)
        # yatay dikey için orano degişstir 12/8, x ve y yi degiştir, xlim degiştir, eli5 score etiketi taşı
        # sns.move_legend(g, "center right")
        # g.set_xticklabels(rotation=90)
        # g.set(ylim=(0, 0.3))
        g.set(xlim=(0, 0.3))
        plt.tight_layout()
        g.set_axis_labels(" Feature Importance Score - "+location_name[i], "")
        # sns.move_legend(g, "upper left", bbox_to_anchor=(.55, .45))
        g.savefig("./Figures/Eli5/eli5_scores_"+scat_name+".png", dpi=300,bbox_inches='tight')
        g.savefig("./Figures/Eli5/eli5_scores_"+scat_name+".eps", format='eps',bbox_inches='tight')

def PlotExtremeWTM_Whole(thr_WTM=20,thr_ERA=0):
    location_name = ["REESE", "MACY", "FLUVANNA"]
    if not os.path.exists("./Figures/PeakWTM_ERA_Whole/"):
        os.makedirs("./Figures/PeakWTM_ERA_Whole/")

    sns.set(style="white", font_scale=1)
    for i, scat_name in enumerate(["Rees", "Macy", "Fluvanna"]):
        INPUT_PATH = "./DATA/ERA5_" + scat_name + "_merged_time_coded.csv"

        """Load data and split"""
        data = pd.read_csv(INPUT_PATH)
        data = data.drop(data[data["max"] < 0].index.tolist(), axis=0)
        total_sample_number=len(data)
        data = data[data["max"]>=thr_WTM]
        data = data[data["ERA5_GUST_10m"]>=thr_ERA]

        data = data.drop("Year", axis=1)
        data = data.drop("Month", axis=1)
        data = data.drop("Day", axis=1)
        data = data.drop("Hour", axis=1)
        correlation = data.corr(method='pearson')
        correlation.to_csv('./Figures/PeakWTM_ERA_Whole/correlation_'+scat_name+'.csv')

        # plt.scatter(np.arange(0,len(data["ERA5_GUST_10m"])), data["ERA5_GUST_10m"], marker='X', color="red", s=4, label='ERA5_GUST_10m')
        # plt.scatter(np.arange(0,len(data["ERA5_GUST_10m"])), data["max"], marker='X', color="blue", s=4, label='WTM - '+location_name[i])
        plt.scatter(x= data["max"], y=data["ERA5_GUST_10m"], marker='X', color="red", s=4, label='ERA5_GUST_10m')

        plt.ylabel('ERA5_GUST_10m')
        plt.xlabel('WTM - '+location_name[i])

        plt.plot()
        plt.title("Total number of sample :"+str(total_sample_number))
        plt.legend()
        plt.ylim([0, 50])
        plt.savefig("./Figures/PeakWTM_ERA_Whole/"+scat_name+"_thr_WTM" + str(thr_WTM)+"_thr_ERA"+str(thr_ERA),bbox_inches='tight')
        plt.savefig("./Figures/PeakWTM_ERA_Whole/"+scat_name+"_thr_WTM" + str(thr_WTM)+"_thr_ERA"+str(thr_ERA)+".eps", format='eps',bbox_inches='tight')
        plt.cla()

def PlotExtremeWTM(thr_WTM=20,thr_ERA=0): # save for 2003 2008 2013
    location_name = ["REESE", "MACY", "FLUVANNA"]
    if not os.path.exists("./Figures/PeakWTM_ERA/"):
        os.makedirs("./Figures/PeakWTM_ERA/")

    sns.set(style="white", font_scale=1.5)
    for i, scat_name in enumerate(["Rees"]):#, "Macy", "Fluvanna",2008,2013
        for year in [2003,2008,2013]:
            INPUT_PATH = "./DATA/ERA5_" + scat_name + "_merged_time_coded.csv"

            """Load data and split"""
            data = pd.read_csv(INPUT_PATH)
            data = data.drop(data[data["max"] < 0].index.tolist(), axis=0)
            total_sample_number=len(data)
            data = data[data["max"]>=thr_WTM]
            data = data[data["i10fg"]>=thr_ERA]

            xlabel="$\mathregular{W_p \: (ms}^{-1})$ - "+location_name[i]
            data = data[data["Year"]==(year-2000)] # for year 2003

            marker_size=20
            plt.figure(figsize=(6,6))
            #2003 ERA5_GUST_10m
            plt.scatter(x= data["max"], y=data["i10fg"], marker='X', color="red", s=marker_size)
            plt.ylabel("$\mathregular{W_{p10^i}\:(ms}^{-1})$")
            plt.xlabel(xlabel)
            plt.xlim([20, 40])
            plt.ylim([0, 40])
            pr2 = pearsonr((data["max"]-np.mean(data["max"])),(data["i10fg"]-np.mean(data["i10fg"])))
            plt.text(37,36, str(year))
            plt.text(35.5, 1, "$\mathregular{r}=$%0.2f" % pr2[0])

            plt.savefig("./Figures/PeakWTM_ERA/"+scat_name+"_"+str(year)+"_thr_WTM" + str(thr_WTM)+"_thr_ERA"+str(thr_ERA)+"_ERA5_GUST_10m_"+"Sample_"+str(len(data)),bbox_inches='tight')
            plt.savefig("./Figures/PeakWTM_ERA/"+scat_name+"_"+str(year)+"_thr_WTM" + str(thr_WTM)+"_thr_ERA"+str(thr_ERA)+"_ERA5_GUST_10m_"+"Sample_"+str(len(data))+".eps", format='eps',bbox_inches='tight')
            plt.cla()

            #2003 ERA5_WSPD_10m
            plt.figure(figsize=(6,6))
            plt.scatter(x= data["max"], y=data["ERA5_WSPD_10m"], marker='X', color="red", s=marker_size)
            plt.ylabel( "$\mathregular{W_{10}\:(ms}^{-1})$")
            plt.xlabel(xlabel)
            plt.xlim([20, 40])
            plt.ylim([0, 40])
            pr2 = pearsonr((data["max"]-np.mean(data["max"])),(data["ERA5_WSPD_10m"]-np.mean(data["ERA5_WSPD_10m"])))
            plt.text(37, 36, str(year))
            plt.text(35.5, 1, "$\mathregular{r}=$%0.2f" % pr2[0])
            plt.savefig("./Figures/PeakWTM_ERA/"+scat_name+"_"+str(year)+"_thr_WTM" + str(thr_WTM)+"_thr_ERA"+str(thr_ERA)+"_ERA5_WSPD_10m_"+"Sample_"+str(len(data)),bbox_inches='tight')
            plt.savefig("./Figures/PeakWTM_ERA/"+scat_name+"_"+str(year)+"_thr_WTM" + str(thr_WTM)+"_thr_ERA"+str(thr_ERA)+"_ERA5_WSPD_10m_"+"Sample_"+str(len(data))+".eps", format='eps',bbox_inches='tight')
            plt.cla()

            #2003 ERA5_CAPE
            plt.figure(figsize=(6,6))
            plt.scatter(x= data["max"], y=data["cape"], marker='X', color="red", s=marker_size)
            plt.ylabel( "$\mathregular{CAPE \: (J\: kg}^{-1})$")
            plt.xlabel(xlabel)
            plt.xlim([20, 40])
            plt.ylim([-50, 3000])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            pr2 = pearsonr((data["max"]-np.mean(data["max"])),(data["cape"]-np.mean(data["cape"])))
            plt.text(37, 2700, str(year))
            plt.text(35.5, 25, "$\mathregular{r}=$%0.2f" % pr2[0])
            plt.savefig("./Figures/PeakWTM_ERA/"+scat_name+"_"+str(year)+"_thr_WTM" + str(thr_WTM)+"_thr_ERA"+str(thr_ERA)+"_ERA5_CAPE_"+"Sample_"+str(len(data)),bbox_inches='tight')
            plt.savefig("./Figures/PeakWTM_ERA/"+scat_name+"_"+str(year)+"_thr_WTM" + str(thr_WTM)+"_thr_ERA"+str(thr_ERA)+"_ERA5_CAPE_"+"Sample_"+str(len(data))+".eps", format='eps',bbox_inches='tight')
            plt.cla()

            #2003 ERA5_CIN
            plt.figure(figsize=(6,6))
            plt.scatter(x= data["max"], y=data["cin"], marker='X', color="red", s=marker_size)
            plt.ylabel( "$\mathregular{CIN \: (J\: kg}^{-1})$")
            plt.xlabel(xlabel)
            plt.xlim([20, 40])
            plt.ylim([-40000, 1000])
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            pr2 = pearsonr((data["max"]-np.mean(data["max"])),(data["cin"]-np.mean(data["cin"])))
            plt.text(37, -3000, str(year))
            plt.text(35.5, -39000, "$\mathregular{r}=$%0.2f"% pr2[0])
            plt.savefig("./Figures/PeakWTM_ERA/"+scat_name+"_"+str(year)+"_thr_WTM" + str(thr_WTM)+"_thr_ERA"+str(thr_ERA)+"_ERA5_CIN_"+"Sample_"+str(len(data)),bbox_inches='tight')
            plt.savefig("./Figures/PeakWTM_ERA/"+scat_name+"_"+str(year)+"_thr_WTM" + str(thr_WTM)+"_thr_ERA"+str(thr_ERA)+"_ERA5_CIN_"+"Sample_"+str(len(data))+".eps", format='eps',bbox_inches='tight')
            plt.cla()


def VisualizeTreeModel():
    # html yazi sorununu çözmek için features lari degiştirip html içerisinden düzenle
    if not os.path.exists("./Figures/DTree/"):
        os.makedirs("./Figures/DTree/")
    for scat_name in ["Macy"]:#"Rees",,"Fluvanna"
        INPUT_PATH="./DATA/ERA5_"+scat_name+"_merged_time_coded.csv"

        for year in range(2003, 2004, 1):
            dataset=DataSet(INPUT_PATH,train_year=year,feature_sel=0)

            X_train, y_train, X_val, y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
            x_train=np.concatenate((X_train,X_val))
            y_train=np.concatenate((y_train,y_val))

            sample_num=40
            # start=4850
            start=0
            # start=200
            #decision tree
            dt=DecisionTreeRegressor(max_depth=2,min_samples_leaf=10)
            dt.fit(x_train[start:(start+sample_num),:],y_train[start:(start+sample_num)])
            cs = {
                      'scatter_edge': "black",
                      'split_line': "black",
                      'mean_line': '#f46d43',
                      'axis_label': "black",
                      'title': "black",
                      'legend_title': "black",
                      'legend_edge': "black",
                      'edge': "black",
                      'rect_edge': "black",
                      'text': "black",
                      'arrow': "black",
                      'node_label': "black",
                      'tick_label': "black",
                      'leaf_label': "black",
                      }
            features = [
                # "W<sub>10</sub>",
                # "W<sub>100</sub>",
                # "W<sub>i10</sub>",
                # "        &alpha;",
                "$\mathregular{W}_{10}$",
                "$\mathregular{W}_{100}$",
                "$\mathregular{W}_{i10}$",
                "${\\alpha}$",
                "$\mathregular{\\beta}$",
                "$\mathregular{T}_{2}$",
                "$\mathregular{T}_{0}$",
                "$\mathregular{T}_{s}$",
                "$\mathregular{T}_{d2}$",
                "$\mathregular{\\Delta}\: {T}_{1}$",
                "$\mathregular{\\Delta}\: {T}_{2}$",
                "$\mathregular{\\Delta}\: {T}_{3}$",
                "$\mathregular{u}_{*}$",

                "$\mathregular{H}_{S}$",
                "$\mathregular{H}_{L}$",
                "$\mathregular{P}_{0}$",
                "$\mathregular{H}$",
                "TCC", "LCC",
                "$\mathregular{\\overline{\\varepsilon}}$",

                "CAPE", "CIN",
                "Month","Hour","Day"
                "HRSin", "HRCos", "DYSin", "DYCos", "MOSin", "MOCos",

            ]
            viz = dtreeviz(dt, X_train[:sample_num,:], y_train[:sample_num], target_name="$\mathregular{W_p \: (ms}^{-1})$",  # this name will be displayed at the leaf node
                           feature_names=features,   fontname="Arial",
                           X=X_train[sample_num,:],
                            orientation ='LR',#title=scat_name +": "+str(year),
                           title_fontsize=14,ticks_fontsize=12,label_fontsize=14,colors=cs
                           ) #fancy=False,colors={ 'scatter_marker': '#ed7f31'},
            viz.view()
            a = 4

def PlotConfusionMatrix(threshold=20):
    location_name = ["REESE", "MACY", "FLUVANNA"]
    if not os.path.exists("./Figures/ConfusionMatrix/"):
        os.makedirs("./Figures/ConfusionMatrix/")
    for year in range(2003,2004):
        for model_name in  ["XGB", "RF"]:
            for i, scat_name in enumerate(["Rees", "Macy", "Fluvanna"]):
                INPUT_PATH = "./DATA/ERA5_" + scat_name + "_merged_time_coded.csv"
                dataset = DataSet(INPUT_PATH, train_year=year, feature_sel=0)

                X_train, y_train, X_val, y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
                y_train_c, y_val_c, y_test_c=converToClassValue(dataset, X_train, y_train,X_val,y_val, X_test, y_test,threshold)
                for im, est in enumerate(['xgboost', "rf"]):  # , "rf", "lgbm"
                    modelFolder = "./Models/"+scat_name+"_models_" + str(dataset.train_year) + "/" + est + '/automl.pkl'

                    automl = pickle.load(open(modelFolder, 'rb'))
                    y_pred = automl.predict(X_test)
                    y_pred = dataset.UnScaleData(X_test, y_pred)
                    y_pred_c=y_pred>threshold
                    cm = confusion_matrix(y_test_c, y_pred_c)
                    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)

                    sns.set_theme()
                    sns.heatmap(cm, annot=True, cmap='binary',cbar=False)

                    # disp.plot(include_values=True, cmap="binary", xticks_rotation='horizontal', values_format=None, ax=None, colorbar=False, transparent=True)
                    # disp.plot()
                    py.savefig("./Figures/ConfusionMatrix/" + scat_name + "_" + model_name+"_"+str(threshold) + "_" + str(year) + "_qq_plot.png",dpi=300, bbox_inches='tight')
                    py.savefig("./Figures/ConfusionMatrix/" + scat_name + "_" + model_name+"_"+str(threshold) + "_" + str(year) + "_qq_plot.eps",format='eps', bbox_inches='tight')
                    py.cla()


plotFigure3()