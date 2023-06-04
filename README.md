# PeakWindGustEstimation
This repository implements the paper ["A decision tree-based measure-correlate-predict approach for peak wind gust estimation from a global reanalysis dataset"](https://wes.copernicus.org/preprints/wes-2023-30/#discussion) 

## Abstract
Peak wind gust ($W_p$) is a crucial meteorological variable for wind farm planning and operations. However, for many wind farm sites, there is a dearth of on-site measurements of $W_p$. In this paper, we propose a machine-learning approach (called INTRIGUE) that utilizes numerous inputs from a public-domain reanalysis dataset, and in turn, generates long-term, site-specific  $W_p$ series. Through a systematic feature importance study, we also identify the most relevant meteorological variables for $W_p$ estimation. Even though the proposed INTRIGUE approach performs very well for nominal conditions compared to specific baselines, its performance for extreme conditions is less than satisfactory
 
## Study Area
This study focuses on the West Texas Panhandle region, one of the largest semi-arid regions in the world.

![Study Area](/docs/StudyArea.PNG)


## Requirements
- Flaml 1.0.1
- Python 3.7
- eli =0.13.0 
- Others...  Please check the requirements.txt file

## Introduction 
In this study, we proposed a decision tree-based MCP approach (called INTRIGUE) for peak wind gust estimation. This approach utilizes several meteorological variables (including the instantaneous wind gust variable) from the ERA5 reanalysis dataset as input features. For non-extreme (i.e., nominal) cases, the INTRIGUE approach-predicted peak wind gust values are closer to the observed ones than the baseline approaches. This approach can also make predictions for neighboring stations where training data is not available. 
 

## Example Results
Confusion matrices for extreme wind gust ($W_p > 20$ m~s$^{-1}$) prediction. The top and bottom panels represent XGBoost and RF models, respectively. The left, middle, and right panels correspond to REESE, MACY, and FLUVANNA stations, respectively.
![Results](/docs/results.JPG)

## Run
1. If you want to retrain models, first go "plant_detection_models",  (**Skip this step if you want to work with the pre-trained model**)
   - Training 
     - Run python model_main.py --alsologtostderr --model_dir=training/ --pipeline_config_path=training/[model name]
     - For Example: python model_main.py --alsologtostderr --model_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config


   - Create inference grap
     - For example: python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-20000 --output_directory inference_graph
   - move the created "inference graph"  file under the "Plant_Detection" directory 

2. Testing
   - Execute "Plant_Detection" folder 
   - Select  "sample_mungbean_ply" file (file with raw point cloud mung bean files) via user interface
   - Press "Start Plant Detection" button
   ![User Interface](/docs/user_interface.JPG)


## How to cite
Kartal, S., Basu, S., and Watson, S. J.: A decision tree-based measure-correlate-predict approach for peak wind gust estimation from a global reanalysis dataset, Wind Energ. Sci. Discuss. [preprint], https://doi.org/10.5194/wes-2023-30, in review, 2023.


