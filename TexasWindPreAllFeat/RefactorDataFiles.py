import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import warnings
import pickle
from TexasData import DataSet
from sklearn.preprocessing import  MinMaxScaler,StandardScaler
from Parameters import *
warnings.simplefilter(action='ignore', category=FutureWarning)
# Last updated: April 20 2022 (Serkan KARTAL)

# *****************************************************************
# 1     Array ID: 1 (observations taken every 5 minutes)
# 2     Julian Day
# 3     Local Standard Time (add one hour for DAYLIGHT savings time)
#       CST = UTC - 6  CDT = UTC - 5
# 4     Station ID
# 5     10-meter Wind Speed (Scalar value in m/sec)=5-minute average of 3-sec values
# 6     10-meter Wind Speed (Vector value in m/sec)=5-minute average of 3-sec values
# 7     10-meter Wind Direction in degrees=5-minute average of 3-sec values
# 8     10-meter Wind Direction-STD Deviation
# 9     10-meter Wind Speed-STD Deviation
# 10    10-meter Wind Speed-Peak 3-second wind gust (m/sec)
# 11    1.5-meter Temperature in degrees C
# 12    9-meter Temperature (Heat Flux) in degrees C
# 13    2-meter Temperature (Heat Flux) in degrees C
# 14    1.5-meter Relative Humidity
# 15    Station Pressure in mb:  Add 600 to get correct value
# 16    Rainfall in inches (total of 5-minute ob period)
# 17    Dewpoint in degrees C=5-minute average of 3-sec values
# 18    2-meter Wind Speed in m/sec=5-minute average of 3-sec values
# *****************************************************************

# Your Input----------------------------------------------------------------
# This is where the raw data is stored on my computer (you need to change
# it appropriately)
DATA_DIR = "./DATA/WTM_"+station

# This is where the extracted data will be stored (you need to change
# it appropriately)
EXT_DATA_DIR = "./DATA/TEMP_"+station+"/"
if not os.path.exists(EXT_DATA_DIR):
    os.makedirs(EXT_DATA_DIR)
# End Your Input------------------------------------------------------------

# The extracted data will be written here:
EXT_DATA_FILE = EXT_DATA_DIR +"SNYD_WTM.txt"
id1_headers= ["ArrayId","Day","Time","StationId","10_meterWindSpeedScalar","10_meterWindSpeedvector","10_meterWindDrectionDeg5",
              "10_meterWindDirectionSTD","10_meterWindSpeedSTD","10_meterWindSpeedPeak","1.5_meterTemp","9_meterTemp","2_meterTemp",
              "1.5_meter_Relative_Humidity","StationPressure","Rainfall","Dewpoint","2_meterWindSpeed"]
id1_headers_new= ["ArrayId","Day","Year","Month","Hour","Min","StationId","10_meterWindSpeedScalar","10_meterWindSpeedvector","10_meterWindDrectionDeg5",
              "10_meterWindDirectionSTD","10_meterWindSpeedSTD","10_meterWindSpeedPeak","1.5_meterTemp","9_meterTemp","2_meterTemp",
              "1.5_meter_Relative_Humidity","StationPressure","Rainfall","Dewpoint","2_meterWindSpeed"]
id2_headers= ["ArrayId","Day","Time","StationId","NaturalSoilTemp5cm","NaturalSoilTemp10cm","NaturalSoilTemp20cm",
              "BareSoilTemp5cm","BareSoilTemp10cm","BareSoilTemp20cm","WaterContentReflectometer5cm",
              "WaterContentReflectometer20cm","WaterContentReflectometer60cm","WaterContentReflectometer70cm",
              "LeafWetnessSensor","BatteryVoltage"]

def seperateIDs5Min(zone_name,selectedIds,mergeType=1):#0 use last one 1 avg 2 max
    File_List = os.listdir(DATA_DIR)

    for id in selectedIds:
        if not os.path.exists(EXT_DATA_DIR + "/" + str(id)):
            os.makedirs(EXT_DATA_DIR + "/" + str(id))

    File_List.sort()

    for i, datafile in enumerate(File_List):
        print(datafile)

        df_new = pd.DataFrame(columns=id1_headers_new)
        if datafile.__contains__(zone_name):
            df = pd.read_csv(DATA_DIR + "/" + datafile, sep=",", header=None)
            for id in selectedIds:
                pd_Data = df[df[0] == id]
                pd_Data = pd_Data.dropna(axis=1)
                if pd_Data.shape[1] > 18:  # datafile.__contains__("REES0508"):#
                    pd_Data = pd_Data.loc[:, :17]

                Year = int(datafile[4:6])
                Month = int(datafile[6:8])

                if id == 1:
                    pd_Data.columns = id1_headers
                elif id == 2:
                    pd_Data.columns = id2_headers

                for i in range(pd_Data.shape[0]):
                    line=pd_Data.iloc[i].copy()
                    time="{:04d}".format(int(line["Time"]))
                    hour = int(time[:2])
                    min = int(time[2:])

                    line.loc["Year"] = Year
                    line["Month"] = Month
                    line["Hour"] = hour
                    line["Min"] = min
                    line=line.drop(['Time'])

                    # temp=temp.append(line)
                    df_new =pd.concat([df_new, line.to_frame().T])

        pd.options.display.float_format = '${:,.2f}'.format
        df_new.to_csv(EXT_DATA_DIR + "/" + str(id) + "/" + datafile, index=False)

def CreateTimeAtTexas(row):
    startDate = datetime(int(row['Year']) + 2000, month=1, day=1)
    endDate = startDate + timedelta(days=int(row['Day'] - 1), hours=int(row['Hour']), minutes=int(row['Min']))
    return endDate

def ConvertTexasData2DateTimeFormat_Apply_Sep():
    File_List = os.listdir(EXT_DATA_DIR+ "/1/")

    if not os.path.exists("./DATA/"+station+"_time/"):
            os.makedirs("./DATA/"+station+"_time/")

    for i, datafile in enumerate(File_List):
        df_texas = pd.read_csv(EXT_DATA_DIR + "/1/" + datafile, sep=",")
        df_texas['TIME'] = df_texas.apply(CreateTimeAtTexas, axis=1)

        df_texas=df_texas.drop(columns=["ArrayId","StationId","10_meterWindSpeedScalar","10_meterWindSpeedvector","10_meterWindDrectionDeg5",
                  "10_meterWindDirectionSTD","10_meterWindSpeedSTD","1.5_meterTemp","9_meterTemp",
                  "1.5_meter_Relative_Humidity","StationPressure","Rainfall","Dewpoint","2_meterWindSpeed"])

        df_texas.to_csv("./DATA/"+station+"_time/" + datafile, index=False)

def ConvertTexasFromMin2Hour_Sep(time_range=1, padding=30):  # time_range hour, padding min, 3 önce 3 sonra alır 3 sonra 00 05 10
    cols=["Day","Year","Month","Hour","TIME","2_meterTemp","mean","max","min"]
    INPUT_FOLDER="./Data/"+station+"_time/"
    File_List = os.listdir(INPUT_FOLDER)
    File_List.sort()

    df_new = pd.DataFrame(columns=cols)
    for i, datafile in enumerate(File_List):
        print(datafile)
        data_min = pd.read_csv(INPUT_FOLDER+ datafile, sep=",")
        data_min["TIME"] = pd.to_datetime(data_min["TIME"])
        time_index = data_min["TIME"].min()
        end_date = data_min["TIME"].max()

        if i>0:
            df_temp = pd.read_csv(INPUT_FOLDER+ File_List[i-1], sep=",")
            data_min= pd.concat([data_min, df_temp])
        if i<(len(File_List)-1):
            df_temp = pd.read_csv(INPUT_FOLDER + File_List[i+1], sep=",")
            data_min= pd.concat([data_min, df_temp])
        data_min["TIME"] = pd.to_datetime(data_min["TIME"])

        while time_index < end_date:
            df_hourly=pd.Series(index=cols)

            temp = data_min[data_min["TIME"] > (time_index - timedelta(minutes=padding))]
            temp = temp[temp["TIME"] <= (time_index + timedelta(minutes=padding))]
            middle_index=int(temp.shape[0]/2)

            if temp.shape[0]<12:
                time_index = time_index + timedelta(hours=1)
                continue
            mean=temp.mean().astype(float).round(2)
            df_hourly["Day"]=temp.iloc[middle_index]["Day"]
            df_hourly["Year"]=temp.iloc[middle_index]["Year"]
            df_hourly["Month"]=temp.iloc[middle_index]["Month"]
            df_hourly["Hour"]=temp.iloc[middle_index]["Hour"]
            df_hourly["TIME"]=time_index
            df_hourly["mean"] = mean["10_meterWindSpeedPeak"]
            df_hourly["2_meterTemp"] = mean["2_meterTemp"]

            max=temp.max()
            df_hourly["max"] = max["10_meterWindSpeedPeak"]

            min=temp.min()
            df_hourly["min"] = min["10_meterWindSpeedPeak"]

            df_new = pd.concat([df_new, df_hourly.to_frame().T])
            time_index=time_index + timedelta(hours=1)

    pd.options.display.float_format = '${:,.2f}'.format
    df_new.to_csv("./DATA/"+station+"_hourly_mean_"+str(padding)+".csv", float_format='%.2f', index=False)

def Subtract6Hour(row):
    startDate =pd.to_datetime((row['TIME']))
    endDate = startDate + timedelta(hours=-6)
    return endDate

def Subtract6Hour2TexasTime():
    df_texas = pd.read_csv("./Data/texas_hourly_mean_15.csv", sep=",")
    df_texas['TIME'] = df_texas.apply(Subtract6Hour, axis=1)
    df_texas.to_csv("./DATA/texas_hourly_mean_15_6H.csv", index=False)

def RefactorReesData():
    # seperateIDs5Min("REES", [1]) # ham veriler 3 farklı ID ye sahip 1 olanları ayıklayıp kullanıyor "./DATA/WTM" içerisindeki veriler alınıp sonuçlar  "./DATA/TEMP içerisine yaziliyor
    # ConvertTexasData2DateTimeFormat_Apply_Sep() # DATA/TEMP/ID altındaki ilgile verilere TIME format ekliyor, gereksiz sütunları kaldırıyor
    ConvertTexasFromMin2Hour_Sep() # dakika bazında ayrı olan dosyları alıp, saat olarak istenen şekilde birleştiriyor ve sonucu tek dosya olarak veriyor

    # Subtract6Hour2TexasTime() # era5 veriler 6 saat geriye alınmış doalyısıyla burda bir şey yapmay gerek yok

def drawTimeRangefor5minData(year,month,day,hour,hour_margin=1):
    INPUT_FOLDER = "./Data/WTM_all_time_"+station+"/"
    File_List = os.listdir(INPUT_FOLDER)
    File_List.sort()
    # sns.set_theme()
    filtered_data=None

    INPUT_PATH="./DATA/ERA5_"+station+"_merged_time_coded.csv"
    era5 = pd.read_csv(INPUT_PATH)
    era5["TIME"]=pd.to_datetime(era5["TIME"])

    for i, datafile in enumerate(File_List):
        print(datafile)
        data_min = pd.read_csv(INPUT_FOLDER + datafile, sep=",")


        data_min["TIME"] = pd.to_datetime(data_min["TIME"])
        # data = data_min[data_min["Year"] ==(year-2000)]
        # data = data[data["Month"] ==month]
        # data = data[data["Day"] ==day]
        data=data_min[data_min["TIME"] > str(year) + '-' + str(month) + "-" + str(day)]
        data=data[data["TIME"] < str(year) + '-' + str(month) + "-" + str(day+1)]
        data = data[data["Hour"]>=(hour-hour_margin)]
        data = data[data["Hour"]<(hour+hour_margin)]
        # data=data[data_min["TIME"] == str(year) + '-' + str(month) + "-" + str(day)]

        if data.size == 0:
            continue

        if filtered_data is None:
            filtered_data=data
        else:
            filtered_data = pd.concat([filtered_data, data])
    era5= era5[era5["TIME"] > str(year) + '-' + str(month) + "-" + str(day)]
    era5 = era5[era5["TIME"] < str(year) + '-' + str(month) + "-" + str(day + 1)]

    era5 = era5[era5["Hour"] >= (hour - hour_margin)]
    era5 = era5[era5["Hour"] < (hour + hour_margin)]
    # era5_hours= era5[["TIME","ERA5_GUST_10m"]]
    era5_hours= era5

    sns.set(style="white", font_scale=1)
    plot_data=filtered_data[["TIME","10_meterWindSpeedPeak"]]

    for modelyear in range(2003,2004,1):
        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot('TIME', '10_meterWindSpeedPeak', data=plot_data, marker='o', color='black', markersize=4, label='REESE(5 min)')
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("$\mathregular{W_p \: (ms}^{-1})$")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # plt.scatter(era5_hours["TIME"], era5_hours["ERA5_GUST_10m"], marker='X', color="red", s=44, label='i10fg')
        # plt.scatter(era5_hours["TIME"], era5_hours["ERA5_WSPD_10m"], marker='o', color="red", s=44, label='fg10')
        # plt.scatter(era5_hours["TIME"], era5_hours["ERA5_WSPD_100m"], marker='+', color="red", s=44, label='ERA5_WSPD_100m-5')
        # plt.scatter(era5_hours["TIME"], era5_hours["ERA5_alpha"], marker='8', color="red", s=44,label=None)# label='ERA5_alpha-5')
        # plt.scatter(era5_hours["TIME"], era5_hours["ERA5_beta"], marker='s', color="red", s=44, label=None)# label='ERA5_beta-5')
        # plt.scatter(era5_hours["TIME"], era5_hours["ERA5_UST"], marker='p', color="yellow", s=44, label=None)# label='ERA5_UST-5')
        # plt.scatter(era5_hours["TIME"], era5_hours["ERA5_SHFX"], marker='p', color="green", s=44,  label='ERA5_SHFX-5')
        # plt.scatter(era5_hours["TIME"], era5_hours["ERA5_PMSL"], marker='p', color="blue", s=44,label=None)#  label='ERA5_PMSL-5')
        # plt.scatter(era5_hours["TIME"], era5_hours["ERA5_PBLH"], marker='p', color="yellow", s=44,label=None)#  label='ERA5_PBLH-5')
        # plt.scatter(era5_hours["TIME"], era5_hours["ERA5_TCC"], marker='p', color="blue", s=44, label=None)# label='ERA5_TCC-5')
        # plt.scatter(era5_hours["TIME"], era5_hours["ERA5_LCC"], marker='p', color="green", s=44, label=None)# label='ERA5_LCC-5')
        # plt.scatter(era5_hours["TIME"], era5_hours["ERA5_EDR"], marker='p', color="red", s=44, label=None)# label='ERA5_EDR-5')
        # plt.scatter(era5_hours["TIME"], era5_hours["ERA5_CAPE"], marker='p', color="red", s=44,label=None)#  label='ERA5_CAPE-5')

        #load model ,ERA5_T_2m,ERA5_TSK,ERA5_TSL,ERA5_Td_2m,ERA5_dT1,ERA5_dT2,ERA5_dT3,ERA5_UST,ERA5_SHFX,ERA5_LH,ERA5_PMSL,ERA5_PBLH,ERA5_TCC,ERA5_LCC,ERA5_EDR,ERA5_CAPE,ERA5_CIN,max,Year,Month,Hour,Day,hour_sin,hour_cos,day_sin,day_cos,month_sin,month_cos
        automl_rf = pickle.load(open("./models/"+station+"_models_"+str(modelyear)+"/rf/automl.pkl", 'rb'))
        automl_xgb = pickle.load(open("./models/"+station+"_models_"+str(modelyear)+"/xgboost/automl.pkl", 'rb'))

        INPUT_PATH = "./DATA/ERA5_"+station+"_merged_time_coded.csv"
        dataset = DataSet(INPUT_PATH, train_year=2003, feature_sel=0)
        dataset.GetEra5_Texas_MergedData()

        processed_data = pd.read_csv(INPUT_PATH)
        processed_data["TIME"] = pd.to_datetime(processed_data["TIME"])

        data = processed_data[processed_data["TIME"] > str(year) + '-' + str(month) + "-" + str(day)]
        data = data[data["TIME"] < str(year) + '-' + str(month) + "-" + str(day + 1)]
        data = data[data["Hour"] >= (hour - hour_margin)]
        temp_prc_data = data[data["Hour"] < (hour + hour_margin)]
        processed_data=temp_prc_data.drop("TIME",axis=1)
        processed_data=processed_data.drop("Year",axis=1)
        processed_data = processed_data.drop("Month", axis=1)
        processed_data = processed_data.drop("Day", axis=1)
        processed_data = processed_data.drop("Hour", axis=1)

        processed_data= pd.DataFrame(dataset.scaler.transform(processed_data), columns=dataset.columns)

        unused_feat = ['Set']
        processed_data_y = processed_data[dataset.target].values
        features = [col for col in processed_data.columns if col not in unused_feat + [dataset.target] + ["index"]]
        processed_data = processed_data[features].values

        processed_data_y= dataset.UnScaleData(processed_data,processed_data_y)
        plt.scatter(temp_prc_data["TIME"], processed_data_y, marker='p', color="green", s=44, label='REESE(60 min)')
        y_pred_rf = automl_rf.predict(processed_data)
        y_pred_xgb = automl_xgb.predict(processed_data)
        y_pred_rf = dataset.UnScaleData(processed_data,y_pred_rf)
        y_pred_xgb = dataset.UnScaleData(processed_data,y_pred_xgb)

        plt.scatter(era5_hours["TIME"], era5_hours["i10fg"], marker='X', color="red", s=44, label="$\mathregular{W}_{p10^i}$")
        plt.scatter(era5_hours["TIME"], era5_hours["fg10"], marker='X', color="brown", s=44, label="$\mathregular{W}_{p10^m}$")
        plt.scatter(temp_prc_data["TIME"], y_pred_rf, marker='+', color="blue", s=44, label='RF')
        plt.scatter(temp_prc_data["TIME"], y_pred_xgb, marker='+', color="cyan", s=44, label='XGB')


        plt.ylim([0, 35])
        plt.yticks([0,10,20,30])
        plt.tight_layout()

        plt.legend()
        plt.savefig("./Figures/peakGustPlots/" + str(year)+"_"+str(month)+"_"+str(day)+"_model_"+str(modelyear),bbox_inches='tight')
        plt.savefig("./Figures/peakGustPlots/" + str(year)+"_"+str(month)+"_"+str(day)+"_model_"+str(modelyear)+".eps", format='eps',bbox_inches='tight')

        plt.cla()
    # plt.show()

def drawPeakGustfor5minData(limit=30):
    INPUT_FOLDER = "./Data/WTM_all_time_"+station+"/"
    File_List = os.listdir(INPUT_FOLDER)
    File_List.sort()
    sns.set_theme()

    if not os.path.exists("./data/peakGustPlots/"):
        os.makedirs("./data/peakGustPlots/")
    for i, datafile in enumerate(File_List):
        print(datafile)
        data_min = pd.read_csv(INPUT_FOLDER + datafile, sep=",")
        data_min["TIME"] = pd.to_datetime(data_min["TIME"])
        data_temp=data_min[data_min["10_meterWindSpeedPeak"] > limit]

        if data_temp.size == 0:
            continue
        days=np.unique(data_temp["Day"].to_numpy())

        for day in days:
            data = data_min[data_min["Day"] ==day]

            plot_data=data[["TIME","10_meterWindSpeedPeak"]]

            # Plot
            fig, ax = plt.subplots()
            ax.plot( 'TIME', '10_meterWindSpeedPeak', data=plot_data , marker='o', color='darkorange')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

            # plt.show()
            plt.savefig("./data/peakGustPlots/"+data["TIME"].iloc[0].strftime("%Y_%m_%d"))




# seperateIDs5Min('FLUV',[1])
# ConvertTexasData2DateTimeFormat_Apply_Sep()
# ConvertTexasFromMin2Hour_Sep()