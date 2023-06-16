from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
import torch
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from flaml.ml import sklearn_metric_loss_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.preprocessing import  MinMaxScaler,StandardScaler
import cdsapi
import netCDF4
from eli5.permutation_importance import get_score_importances
import seaborn as sn
import matplotlib.pyplot as plt
import pickle

from sklearn.feature_selection import SelectFromModel
np.random.seed(10)
from multiprocessing import Pool
from matplotlib import pyplot as plt
from Parameters import *
import os
import wget
from pathlib import Path

INPUT_PATH="./DATA/ERA5_"+station+"_merged_time_coded.csv"

class DataSet:
    ##added old because not using anymore
    def __init__(self,path,train_year,target="max",feature_sel=0,train_year_list=None):
        self.DataFilePath = path
        self.target=target
        self.feature_sel=feature_sel
        self.train_year=train_year
        self.train_year_list=train_year_list

    def GetEra5_Texas_MergedData(self):#0 none, 1 ELi5
        np.random.seed(1)
        """Load data and split"""
        train = pd.read_csv(self.DataFilePath)
        train = train.drop(train[train[self.target] < 0].index.tolist(), axis=0)

        train=train.reset_index()
        train=train.drop("index",axis=1)
        train_indices = train[(train.Year == (self.train_year-2000)) & (pd.to_datetime(train["TIME"]).dt.day<25)].index
        valid_indices = train[(train.Year == (self.train_year-2000)) & (pd.to_datetime(train["TIME"]).dt.day>=25)].index
        test_indices= train[(train.Year != (self.train_year-2000)) & (train.Year>0)& (train.Year<15) ].index
        if  "TIME" in train:
            train=train.drop("TIME",axis=1)
            train=train.drop("Month",axis=1)
            train=train.drop("Day",axis=1)
            train=train.drop("Hour",axis=1)


        #draw scatter end
        self.dataset_unscaled_year=train.copy()
        """Network parameters"""
        train = train.drop(["Year"], axis=1)

        if self.feature_sel == 1 and os.path.exists("./eli_feature_scores.txt"):
            feature_scores = np.loadtxt("./eli_feature_scores.txt", delimiter=',')
            threshold = 0.01
            indexes=np.argsort(feature_scores)[-10:]
            features = [train.columns[i] for i in indexes] + ["max"]
            train = train[features]

        # transform data
        self.dataset_unscaled=train.copy()
        self.columns = train.columns
        self.scaler = StandardScaler()
        self.scaler=self.scaler.fit(train)

        # # if station will be predicted with rees network
        #
        # train_r = pd.read_csv(INPUT_PATH)
        # train_r = train_r.drop(train_r[train_r[self.target] < 0].index.tolist(), axis=0)

        # if self.feature_sel == 1:
        #     train_r = train_r[features]
        #
        # train_r = train_r.reset_index()
        # train_r = train_r.drop("index", axis=1)
        # if "TIME" in train_r:
        #     train_r = train_r.drop("TIME", axis=1)
        # self.scaler=self.scaler.fit(train_r)

        # end macy prediction with rees

        train= pd.DataFrame(self.scaler.transform(train), columns=self.columns)
        """Simple preprocessing"""
        categorical_columns = []
        categorical_dims = {}

        for col in train.columns[train.dtypes == object]:
            # print(col, train[col].nunique())
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)

        """Define categorical features for categorical embeddings"""
        unused_feat = ['Set']
        features = [col for col in train.columns if col not in unused_feat + [self.target] + ["index"]]
        cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
        cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
        # define your embedding sizes : here just a random choice
        cat_emb_dim = []

        self.features=features.copy()
        self.target_index = np.where(self.columns.values == self.target)[0][0]
        """Training"""
        X_train = train[features].values[train_indices]
        y_train = train[self.target].values[train_indices].reshape(-1, 1)

        X_val = train[features].values[valid_indices]
        y_val = train[self.target].values[valid_indices].reshape(-1, 1)

        X_test = train[features].values[test_indices]
        y_test = train[self.target].values[test_indices].reshape(-1, 1)

        # self.plotWTM_y_values( y_train)
        return X_train, y_train,X_val,y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs

    def plotWTM_y_values(self,values):
        dataset_Name= self.DataFilePath[12:16]+"_"+str(self.train_year)  #"""./DATA/ERA5_Rees_merged_time_coded.csv"

        if not os.path.exists("./Data/WTM_Images/"):
            os.makedirs("./Data/WTM_Images/")

        fig = plt.figure(figsize=(18, 10))
        x_values=np.arange(values.shape[0])
        plt.plot(x_values,values)
        plt.title(dataset_Name)
        plt.savefig("./Data/WTM_Images/"+dataset_Name+".jpg")
        plt.close("all")

    def GetEra5_Texas_TestData(self,year):
        train_year=self.dataset_unscaled_year.copy()
        train=self.dataset_unscaled.copy()
        test_indices = train[(train_year.Year == (year - 2000))].index

        train = pd.DataFrame(self.scaler.transform(train), columns=self.columns) #close here

        X_test = train[self.features].values[test_indices]
        y_test = train[self.target].values[test_indices].reshape(-1, 1)
        return X_test,y_test

    def ConvertTexasData2DateTimeFormat(self):
        OUTPUT_DIR = "./DATA/"+station+"_merged_time.csv"
        df_texas = pd.read_csv("./DATA/"+station+"_merged.csv")
        df_new = pd.DataFrame()

        for i in range(df_texas.shape[0]):
            row = df_texas.iloc[i]
            startDate = datetime(int(row['Year']) + 2000, month=1, day=1)
            endDate = startDate + timedelta(days=int(row['Day'] - 1), hours=int(row['Hour']))
            new_row = pd.Series(row)
            new_row["TIME"] = endDate
            df_new = df_new.append(new_row)
            if i%1000==0:
                print(i)

        df_new.to_csv(OUTPUT_DIR, index=False)

    def UnScaleData(self,data_x,data_y):
        """Load data and split"""
        data=np.concatenate((data_x[:,:self.target_index],data_y.reshape(data_y.shape[0],-1),data_x[:,self.target_index:]),axis=1)
        unscaled_data=self.scaler.inverse_transform(data)
        return unscaled_data[:,self.target_index]


    def DownloadPart(self,parameters):
        year=parameters["year"]
        month = parameters["month"]
        day=parameters["day"]

        print(str(year[0]) + " " + str(month[0]) + " indiriliyor")
        c = cdsapi.Client()
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis",
            # "area": "33.13/-101.56/33.03/-101.46", #MAcy 33.0815  -101.516
            "area": "33.65/-102.10/33.55/-102",# Rees 33.607 -102.04 # bunlar WTM sayfasından alındı
            'variable': [
                '100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_neutral_wind',
                '10m_u_component_of_wind', '10m_v_component_of_neutral_wind', '10m_v_component_of_wind',
                '10m_wind_gust_since_previous_post_processing', '2m_dewpoint_temperature', '2m_temperature',
                'air_density_over_the_oceans', 'angle_of_sub_gridscale_orography',
                'anisotropy_of_sub_gridscale_orography',
                'benjamin_feir_index', 'boundary_layer_dissipation', 'boundary_layer_height',
                'charnock', 'clear_sky_direct_solar_radiation_at_surface', 'coefficient_of_drag_with_waves',
                'convective_available_potential_energy', 'convective_inhibition', 'convective_precipitation',
                'convective_rain_rate', 'convective_snowfall', 'convective_snowfall_rate_water_equivalent',
                'downward_uv_radiation_at_the_surface', 'duct_base_height', 'eastward_gravity_wave_surface_stress',
                'eastward_turbulent_surface_stress', 'evaporation', 'forecast_albedo',
                'forecast_logarithm_of_surface_roughness_for_heat', 'forecast_surface_roughness',
                'free_convective_velocity_over_the_oceans',
                'friction_velocity', 'geopotential', 'gravity_wave_dissipation',
                'high_vegetation_cover', 'ice_temperature_layer_1', 'ice_temperature_layer_2',
                'ice_temperature_layer_3', 'ice_temperature_layer_4', 'instantaneous_10m_wind_gust',
                'instantaneous_eastward_turbulent_surface_stress',
                'instantaneous_large_scale_surface_precipitation_fraction', 'instantaneous_moisture_flux',
                'instantaneous_northward_turbulent_surface_stress', 'instantaneous_surface_sensible_heat_flux',
                'k_index',
                'lake_bottom_temperature', 'lake_cover', 'lake_depth',
                'lake_ice_depth', 'lake_ice_temperature', 'lake_mix_layer_depth',
                'lake_mix_layer_temperature', 'lake_shape_factor', 'lake_total_layer_temperature',
                'land_sea_mask', 'large_scale_precipitation', 'large_scale_precipitation_fraction',
                'large_scale_rain_rate', 'large_scale_snowfall', 'large_scale_snowfall_rate_water_equivalent',
                'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation', 'low_vegetation_cover',
                'maximum_2m_temperature_since_previous_post_processing', 'maximum_individual_wave_height',
                'maximum_total_precipitation_rate_since_previous_post_processing',
                'mean_boundary_layer_dissipation', 'mean_convective_precipitation_rate',
                'mean_convective_snowfall_rate',
                'mean_direction_of_total_swell', 'mean_direction_of_wind_waves',
                'mean_eastward_gravity_wave_surface_stress',
                'mean_eastward_turbulent_surface_stress', 'mean_evaporation_rate', 'mean_gravity_wave_dissipation',
                'mean_large_scale_precipitation_fraction', 'mean_large_scale_precipitation_rate',
                'mean_large_scale_snowfall_rate',
                'mean_northward_gravity_wave_surface_stress', 'mean_northward_turbulent_surface_stress',
                'mean_period_of_total_swell',
                'mean_period_of_wind_waves', 'mean_potential_evaporation_rate', 'mean_runoff_rate',
                'mean_sea_level_pressure', 'mean_snow_evaporation_rate', 'mean_snowfall_rate',
                'mean_snowmelt_rate', 'mean_square_slope_of_waves', 'mean_sub_surface_runoff_rate',
                'mean_surface_direct_short_wave_radiation_flux',
                'mean_surface_direct_short_wave_radiation_flux_clear_sky',
                'mean_surface_downward_long_wave_radiation_flux',
                'mean_surface_downward_long_wave_radiation_flux_clear_sky',
                'mean_surface_downward_short_wave_radiation_flux',
                'mean_surface_downward_short_wave_radiation_flux_clear_sky',
                'mean_surface_downward_uv_radiation_flux', 'mean_surface_latent_heat_flux',
                'mean_surface_net_long_wave_radiation_flux',
                'mean_surface_net_long_wave_radiation_flux_clear_sky', 'mean_surface_net_short_wave_radiation_flux',
                'mean_surface_net_short_wave_radiation_flux_clear_sky',
                'mean_surface_runoff_rate', 'mean_surface_sensible_heat_flux',
                'mean_top_downward_short_wave_radiation_flux',
                'mean_top_net_long_wave_radiation_flux', 'mean_top_net_long_wave_radiation_flux_clear_sky',
                'mean_top_net_short_wave_radiation_flux',
                'mean_top_net_short_wave_radiation_flux_clear_sky', 'mean_total_precipitation_rate',
                'mean_vertical_gradient_of_refractivity_inside_trapping_layer',
                'mean_vertically_integrated_moisture_divergence', 'mean_wave_direction',
                'mean_wave_direction_of_first_swell_partition',
                'mean_wave_direction_of_second_swell_partition', 'mean_wave_direction_of_third_swell_partition',
                'mean_wave_period',
                'mean_wave_period_based_on_first_moment', 'mean_wave_period_based_on_first_moment_for_swell',
                'mean_wave_period_based_on_first_moment_for_wind_waves',
                'mean_wave_period_based_on_second_moment_for_swell',
                'mean_wave_period_based_on_second_moment_for_wind_waves', 'mean_wave_period_of_first_swell_partition',
                'mean_wave_period_of_second_swell_partition', 'mean_wave_period_of_third_swell_partition',
                'mean_zero_crossing_wave_period',
                'minimum_2m_temperature_since_previous_post_processing',
                'minimum_total_precipitation_rate_since_previous_post_processing',
                'minimum_vertical_gradient_of_refractivity_inside_trapping_layer',
                'model_bathymetry', 'near_ir_albedo_for_diffuse_radiation', 'near_ir_albedo_for_direct_radiation',
                'normalized_energy_flux_into_ocean', 'normalized_energy_flux_into_waves',
                'normalized_stress_into_ocean',
                'northward_gravity_wave_surface_stress', 'northward_turbulent_surface_stress',
                'ocean_surface_stress_equivalent_10m_neutral_wind_direction',
                'ocean_surface_stress_equivalent_10m_neutral_wind_speed', 'peak_wave_period',
                'period_corresponding_to_maximum_individual_wave_height',
                'potential_evaporation', 'precipitation_type', 'runoff',
                'sea_ice_cover', 'sea_surface_temperature', 'significant_height_of_combined_wind_waves_and_swell',
                'significant_height_of_total_swell', 'significant_height_of_wind_waves',
                'significant_wave_height_of_first_swell_partition',
                'significant_wave_height_of_second_swell_partition', 'significant_wave_height_of_third_swell_partition',
                'skin_reservoir_content',
                'skin_temperature', 'slope_of_sub_gridscale_orography', 'snow_albedo',
                'snow_density', 'snow_depth', 'snow_evaporation',
                'snowfall', 'snowmelt', 'soil_temperature_level_1',
                'soil_temperature_level_2', 'soil_temperature_level_3', 'soil_temperature_level_4',
                'soil_type', 'standard_deviation_of_filtered_subgrid_orography', 'standard_deviation_of_orography',
                'sub_surface_runoff', 'surface_latent_heat_flux', 'surface_net_solar_radiation',
                'surface_net_solar_radiation_clear_sky', 'surface_net_thermal_radiation',
                'surface_net_thermal_radiation_clear_sky',
                'surface_pressure', 'surface_runoff', 'surface_sensible_heat_flux',
                'surface_solar_radiation_downward_clear_sky', 'surface_solar_radiation_downwards',
                'surface_thermal_radiation_downward_clear_sky',
                'surface_thermal_radiation_downwards', 'temperature_of_snow_layer', 'toa_incident_solar_radiation',
                'top_net_solar_radiation', 'top_net_solar_radiation_clear_sky', 'top_net_thermal_radiation',
                'top_net_thermal_radiation_clear_sky', 'total_column_ozone', 'total_column_rain_water',
                'total_column_snow_water', 'total_column_supercooled_liquid_water', 'total_column_water',
                'total_column_water_vapour', 'total_precipitation', 'total_sky_direct_solar_radiation_at_surface',
                'total_totals_index', 'trapping_layer_base_height', 'trapping_layer_top_height',
                'type_of_high_vegetation', 'type_of_low_vegetation', 'u_component_stokes_drift',
                'uv_visible_albedo_for_diffuse_radiation', 'uv_visible_albedo_for_direct_radiation',
                'v_component_stokes_drift',
                'vertical_integral_of_divergence_of_cloud_frozen_water_flux',
                'vertical_integral_of_divergence_of_cloud_liquid_water_flux',
                'vertical_integral_of_divergence_of_geopotential_flux',
                'vertical_integral_of_divergence_of_kinetic_energy_flux',
                'vertical_integral_of_divergence_of_mass_flux', 'vertical_integral_of_divergence_of_moisture_flux',
                'vertical_integral_of_divergence_of_ozone_flux',
                'vertical_integral_of_divergence_of_thermal_energy_flux',
                'vertical_integral_of_divergence_of_total_energy_flux',
                'vertical_integral_of_eastward_cloud_frozen_water_flux',
                'vertical_integral_of_eastward_cloud_liquid_water_flux',
                'vertical_integral_of_eastward_geopotential_flux',
                'vertical_integral_of_eastward_heat_flux', 'vertical_integral_of_eastward_kinetic_energy_flux',
                'vertical_integral_of_eastward_mass_flux',
                'vertical_integral_of_eastward_ozone_flux', 'vertical_integral_of_eastward_total_energy_flux',
                'vertical_integral_of_eastward_water_vapour_flux',
                'vertical_integral_of_energy_conversion', 'vertical_integral_of_kinetic_energy',
                'vertical_integral_of_mass_of_atmosphere',
                'vertical_integral_of_mass_tendency', 'vertical_integral_of_northward_cloud_frozen_water_flux',
                'vertical_integral_of_northward_cloud_liquid_water_flux',
                'vertical_integral_of_northward_geopotential_flux', 'vertical_integral_of_northward_heat_flux',
                'vertical_integral_of_northward_kinetic_energy_flux',
                'vertical_integral_of_northward_mass_flux', 'vertical_integral_of_northward_ozone_flux',
                'vertical_integral_of_northward_total_energy_flux',
                'vertical_integral_of_northward_water_vapour_flux',
                'vertical_integral_of_potential_and_internal_energy',
                'vertical_integral_of_potential_internal_and_latent_energy',
                'vertical_integral_of_temperature', 'vertical_integral_of_thermal_energy',
                'vertical_integral_of_total_energy',
                'vertically_integrated_moisture_divergence', 'volumetric_soil_water_layer_1',
                'volumetric_soil_water_layer_2',
                'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4', 'wave_spectral_directional_width',
                'wave_spectral_directional_width_for_swell', 'wave_spectral_directional_width_for_wind_waves',
                'wave_spectral_kurtosis',
                'wave_spectral_peakedness', 'wave_spectral_skewness', 'zero_degree_level',
            ],
            "year": year,
            "month": month,
            # "day": ["01"],
            # "day": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11",
            #         "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22",
            #         "23", "24", "25", "26", "27", "28", "29", "30", "31"],
            "day":day,
             "time": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11",
             "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"],
            'format': 'netcdf',
        }, "./Data/Era5_"+station+"/ERA5_"+str(year[0])+"_"+str(month[0])+"_"+str(day[0])+".nc")

    def DownloadEra5Data(self):
        #
        for year in range(2005,2006,1):
            for month in range(1,13,1):
                year_list=[]
                month_list=[]
                year_list.append(year)
                month_list.append(month)
                # self.DownloadPart(year_list,month_list,day=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"])
                self.DownloadPart(year_list,month_list,day=["12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"])
                # self.DownloadPart(year_list,month_list,day=["23", "24", "25", "26", "27", "28", "29", "30", "31"])
                print(str(year)+" "+str(month)+ "isleniyor")
                # self.PreprocessEra5Data(year,month,day="01")
        # self.MergeExtractedEra5Data()
        # self.MergeTexasData_Era5()

    def DownloadEra5DataParallel(self):
        #
        data_dict_list = []
        for year in range(2003,2014,1):
        # for year in [2011]:
            for month in range(1,13,1):
                year_list=[]
                month_list=[]
                year_list.append(year)
                month_list.append(month)
                # print(str(year)+" "+str(month)+ "  indiriliyor")
                day1=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
                day2=["12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"]
                day3=["23", "24", "25", "26", "27", "28", "29", "30", "31"]
        #         data_dict_list.append({"year": year_list,"month": month_list, "day": day1 })
        #         data_dict_list.append({"year": year_list,"month": month_list, "day": day2 })
        #         data_dict_list.append({"year": year_list,"month": month_list, "day": day3 })

                self.PreprocessEra5Data(year,month,day="01")
                self.PreprocessEra5Data(year, month, day="12")
                self.PreprocessEra5Data(year, month, day="23")
        self.MergeExtractedEra5Data()
        self.MergeTexasData_Era5()

        p = Pool(10)
        # p.map(self.DownloadPart, data_dict_list)

    def PreprocessEra5Data(self,year,month,day):

        columns=[]
        INPUT_DIR = "./Data/Era5_"+station+"/ERA5_"+str(year)+"_"+str(month)+"_"+day+".nc"
        OUTPUT_DIR = "./DATA/ExtractedEra5_"+station+"/"+str(year)+"_"+str(month)+"_"+day+".csv"

        """Load Surface Data"""
        D_SFC = netCDF4.Dataset(INPUT_DIR , 'r')

        i=0
        Combo = None
        for key in D_SFC.variables:
            var=np.squeeze(D_SFC.variables[key][:])
            if var.size == 1:
                continue
            if Combo is None:
                Combo= np.zeros((var.shape[0], 260))
            Combo[:, i] = var[:]
            columns.append(key)
            i=i+1

        U10 = np.squeeze(D_SFC.variables['u10'][:])
        V10 = np.squeeze(D_SFC.variables['v10'][:])
        M10 = np.sqrt(U10 ** 2 + V10 ** 2)

        U100 = np.squeeze(D_SFC.variables['u100'][:])
        V100 = np.squeeze(D_SFC.variables['v100'][:])
        M100 = np.sqrt(U100 ** 2 + V100 ** 2)

        # G10 = np.squeeze(D_SFC.variables['i10fg'][:])  # Gust
        alpha = np.log(M100 / M10) / np.log(10)
        RtoD = 180 / np.pi

        X10 = np.arctan2(-U10, -V10) * RtoD
        II = np.where(X10 < 0)
        X10[II] = X10[II] + 360

        X100 = np.arctan2(-U100, -V100) * RtoD
        II = np.where(X100 < 0)
        X100[II] = X100[II] + 360

        beta = np.abs(X100 - X10)
        II = np.where(beta > 180)
        beta[II] = 360 - beta[II]

        """Merge data"""
        Combo[:,i+ 0] = M10
        Combo[:,i+ 1] = M100
        # Combo[:,i+ 2] = G10

        Combo[:,i+ 2] = alpha
        Combo[:,i+ 3] = beta

        columns.append('ERA5_WSPD_10m')
        columns.append('ERA5_WSPD_100m')
        # columns.append('ERA5_GUST_10m')
        columns.append('ERA5_alpha')
        columns.append('ERA5_beta')

        """Output CSV file"""
        df = pd.DataFrame(data=Combo, columns=columns)
        df = df.rename(columns={'time': 'TIME'})
        day_i= int(day)
        if day_i<20:
            df['TIME'] = pd.date_range(start=str(year)+'-'+str(month)+'-'+str(day_i), end=str(year)+'-'+str(month)+'-'+str(day_i+11), freq='h')[:-1]
        else:
            if month<12:
                df['TIME'] = pd.date_range(start=str(year)+'-'+str(month)+'-'+str(day_i), end=str(year)+'-'+str(month+1), freq='h')[:-1]
            else:
                df['TIME'] = pd.date_range(start=str(year) + '-' + str(month)+'-'+str(day_i), end=str(year + 1) + '-' + str(1),freq='h')[:-1]
        df['TIME'] =df['TIME'] + timedelta(hours=-6)

        # df = df[['TIME', 'ERA5_WSPD_10m', 'ERA5_WSPD_100m', 'ERA5_GUST_10m', 'ERA5_alpha', 'ERA5_beta', 'ERA5_T_2m',
        #          'ERA5_TSK', 'ERA5_TSL', 'ERA5_Td_2m', 'ERA5_dT1', 'ERA5_dT2', 'ERA5_dT3', 'ERA5_UST', 'ERA5_SHFX',
        #          'ERA5_LH', 'ERA5_PMSL', 'ERA5_PBLH', 'ERA5_TCC', 'ERA5_LCC', 'ERA5_EDR', 'ERA5_CAPE', 'ERA5_CIN']]
        # print(df.describe())
        df.to_csv(OUTPUT_DIR , index=False)

    def MergeExtractedEra5Data(self):
        INPUT_DIR = "./Data/ExtractedEra5_"+station+"/"
        OUTPUT_DIR = "./DATA/Era5_"+station+"_merged.csv"
        plyfile_directory = os.listdir(INPUT_DIR )

        df = pd.DataFrame([])
        for i, file in enumerate(plyfile_directory):
            readed_df=pd.read_csv(INPUT_DIR+file)
            df=df.append(readed_df)
        df.to_csv(OUTPUT_DIR, index=False)

    def MergeTexasData_Era5(self):
        OUTPUT_DIR="./DATA/ERA5_"+station+"_merged_time_coded.csv"

        df_texas=pd.read_csv("./DATA/"+station+"_hourly_mean_30.csv")
        df_era5=pd.read_csv("./DATA/Era5_"+station+"_merged.csv")
        df_era5["TIME"]=pd.to_datetime(df_era5["TIME"])
        df_texas["TIME"]=pd.to_datetime(df_texas["TIME"])
        df_texas=df_texas[["TIME","max","Year","Month","Hour","Day"]]
        # df_texas=df_texas[["TIME","10_meterWindSpeedPeak","Year","Month","Hour","Day"]]
        df = pd.merge(df_era5, df_texas, on=['TIME'])
        # df['TIME'] = pd.date_range(start=str(year) + '-' + str(month), end=str(year + 1) + '-' + str(1), freq='h')[:-1]
        df=df.sort_values("TIME")

        df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)

        df['day_sin'] = np.sin(2 * np.pi * df['Day'] / 365.0)
        df['day_cos'] = np.cos(2 * np.pi * df['Day'] / 365.0)

        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)


        df.to_csv(OUTPUT_DIR,index=False)

    def DrawFeatureImportance(self,feature_importances_,model_name):
        # summarize feature importance
        w = []
        for i, v in enumerate(feature_importances_):
            # print('Feature: %0d, Score: %.5f' % (i,v))
            w.append(v)
        bars = plt.bar(range(feature_importances_.shape[0]), feature_importances_, label="weights", )
        plt.xlabel("Features")
        plt.ylabel("Weights")
        plt.title("TabNet")

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + .005, np.round(yval, decimals=2))

        xlocs = [i for i in range(feature_importances_.shape[0])]
        xlabs = [i + 1 for i in range(feature_importances_.shape[0])]
        plt.xticks(xlocs, xlabs)
        plt.savefig("./models/"+model_name+"_weights.png")
        plt.close()

    def ConvertHourly2DailyData(self):
        INPUT_DIR="./DATA/ERA5_"+station+"_merged_time.csv"
        OUTPUT_DIR="./DATA/ERA5_"+station+"_merged_daily.csv"

        df = pd.read_csv(INPUT_DIR )
        new_df=df.groupby(['Year','Month','Day'])[df.columns[1:-4]].max()
        # new_df = df.groupby(['Year','Month','Day']).agg({'10_meterWindSpeedPeak': ['mean', 'min', 'max','std']})
        # new_df=new_df.drop(columns="Hour")
        new_df.to_csv(OUTPUT_DIR, index=True)

    def CreateCorrMatrix(self,INPUT_PATH):
        df = pd.read_csv(INPUT_PATH )

        df = df.drop(df[df[self.target] < 0].index.tolist(), axis=0)

        # df=df.drop(columns=["ERA5_WSPD_10m","ERA5_WSPD_100m","ERA5_alpha","ERA5_beta","ERA5_TSK","ERA5_TSL","ERA5_Td_2m",
        #                     "ERA5_dT1","ERA5_dT2","ERA5_dT3","ERA5_UST","ERA5_SHFX","ERA5_LH","ERA5_PMSL","ERA5_PBLH","ERA5_TCC","ERA5_LCC",
        #                     "ERA5_EDR",	"ERA5_CAPE",	"ERA5_CIN"])

        corrMatrix = df.corr()
        corrMatrix=corrMatrix.round(2)
        sn.set(rc={'figure.figsize': (15, 10)})
        sn.set(font_scale=0.8)
        sn.heatmap(corrMatrix, annot=True)
        plt.savefig("./DATA/ERA5_"+station+"_merged_time_corr.png")

    def CreateTimeSeriesDailyData(self, n_in=3,n_out=1):
        INPUT_DIR="./DATA/ERA5_"+station+"_merged_daily.csv"
        df = pd.read_csv(INPUT_DIR )

        column_indexes=[3,4,5,15,22]
        data_x = df.iloc[:, column_indexes]
        data_y = df.iloc[:, 25]
        columns=data_x.columns

        n_vars = 1 if type(data_x) is list else data_x.shape[1]
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(data_x.shift(i))
            names += [('%s(t-%d)' % (columns[j], i-1)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)

        for i in range(0, n_out):
            cols.append(data_y.shift(-i+1))
            names += [data_y.name]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        agg.dropna(inplace=True)

        agg.to_csv("./DATA/ERA5_"+station+"_merged_daily_TimeSeries.csv",index=False)

    def EvaluateERA5_GUST_10m(self,dataset):
        # columns = ['ERA5_WSPD_10m', 'ERA5_WSPD_100m', 'ERA5_GUST_10m', 'ERA5_alpha', 'ERA5_beta',
        #            'ERA5_T_2m', 'ERA5_TSK', 'ERA5_TSL', 'ERA5_Td_2m', 'ERA5_dT1', 'ERA5_dT2',
        #            'ERA5_dT3', 'ERA5_UST', 'ERA5_SHFX', 'ERA5_LH', 'ERA5_PMSL', 'ERA5_PBLH', 'ERA5_TCC',
        #            'ERA5_LCC', 'ERA5_EDR', 'ERA5_CAPE', 'ERA5_CIN'])

        X_train, y_train, X_val,y_val, X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = dataset.GetEra5_Texas_MergedData()
        era5_index=np.where(dataset.columns.values=="ERA5_GUST_10m")[0][0]
        y_pred = X_test[:,era5_index]

        # y_pred=self.UnScaleData(X_test,y_pred)
        # y_test=self.UnScaleData(X_test,y_test)

        r2, mse, mae = dataset.EvaluateResults("ERA5_GUST_10m", X_test, y_test, y_pred)
        f = open("./Data/raw_data_scores/"+str(dataset.train_year)+"_ERA5_GUST_10m_scores.txt", "w")
        f.write("r2=" + str(r2) + '\n')
        f.write("mse=" + str(mse) + '\n')
        f.write("mae=" + str(mae) + '\n')
        f.close()

    def EvaluateResults(self, model_name, X_test, y_test, y_pred):

        # compute different metric values on testing dataset
        y_pred=self.UnScaleData(X_test,y_pred)
        y_test=self.UnScaleData(X_test,y_test)
        r2= 1 - sklearn_metric_loss_score('r2', y_pred, y_test)
        mse= sklearn_metric_loss_score('mse', y_pred, y_test)
        mae= sklearn_metric_loss_score('mae', y_pred, y_test)

        # print("*****************************")
        print(model_name)
        print('r2', '=',r2)
        print('mse', '=',mse)
        print('mae', '=',mae)
        print("*****************************")

        # f = open("./models/" + model_name +"/"+ "_test_scores.txt", "w")
        # f.write("r2=" + str(r2) + '\n')
        # f.write("mse=" + str(mse) + '\n')
        # f.write("mae=" + str(mae) + '\n')
        # f.close()

        return r2,mse,mae

    def PrintBest(self,results, modelFolder):
        max_index=results["r2"].idxmax()
        results = results.reset_index()

        r2 = results.iloc[max_index]["r2"]
        mse = results.iloc[max_index]["mse"]
        mae = results.iloc[max_index]["mae"]
        config= results.iloc[max_index]["config"]

        print("*************Best Result****************")
        print(modelFolder +"  "+ config )
        print('r2', '=', r2)
        print('mse', '=', mse)
        print('mae', '=', mae)
        print("*****************************")

        results = results.append({"config": config, "r2": r2, "mse": mse, "mae": mae}, ignore_index=True)

        results=results.round(3)
        results.to_csv(modelFolder +"_results.csv", index=False)

    def EvaluateERA5_GUST_10m_Yearly(self):
        X_train, y_train,X_val,y_val,  X_test, y_test, cat_dims, cat_emb_dim, cat_idxs = self.GetEra5_Texas_MergedData()
        era5_index=np.where(self.columns.values=="ERA5_GUST_10m")[0][0]

        r2_dic = {}
        mae_dic = {}
        mse_dic = {}

        if not os.path.exists("./Results/"):
            os.makedirs("./Results/")

        for year in range(2003, 2014, 1):
            x_data, y_test = self.GetEra5_Texas_TestData(year) # bu kullanilmadan once normalizasyonu kapat
            y_pred = x_data[:, era5_index]

            r2 = 1 - sklearn_metric_loss_score('r2', y_pred, y_test)
            mse = sklearn_metric_loss_score('mse', y_pred, y_test)
            mae = sklearn_metric_loss_score('mae', y_pred, y_test)

            r2_dic[year] = r2
            mae_dic[year] = mae
            mse_dic[year] = mse

        r2_df=pd.DataFrame.from_dict([r2_dic])
        mae_df=pd.DataFrame.from_dict([mae_dic])
        mse_df=pd.DataFrame.from_dict([mse_dic])
        r2_df.to_csv("./Results/ERA5_r2.csv")
        mae_df.to_csv("./Results/ERA5_mae.csv")
        mse_df.to_csv("./Results/ERA5_mse.csv")



