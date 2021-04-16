# *Landslide Susceptibility Assessment tool* (LSAT)
Firstly you should prepare reclassed landslide factor rasters.They should be same sizes and same resolutions.Prepared raster files must be shown in below.\
IMPORTANT: All parameter files should be in same folder. There should be no files other than factor raster files in this folder. The name of the parameter raster files must begin with "rec". rec_asp, rec_slp etc. Please do not use too long file names.

Landslide file must be polygonal type as shapefile format.\
Also, the area file must be polygonal type as shapefile format.\
The sample data folder is given below:\
https://github.com/apolat2018/LSAT/tree/master/sampla_data \
Factor raster files are in raster folder. Landslides and area file are in vectors folder.


 <p>
  <img width="1024" height="2048" src="https://github.com/apolat2018/LSAT/blob/master/rasters_image.jpg"><p>

### Installing required libraries

LSAT works on ArcGIS software with the Windows platform. Before using the LSAT, the required installations must
be done. Python 2.7 is already installed with ArcGIS 10.4.
The libraries must be compatible with the version of Python
2.7. Pip is recommended for easy installation. If a different
version of python is installed, pip2.7 must be used. Scikitlearn (Pedregosa et al. 2011) is the main library for our
tasks.
 * Numpy (version of 1.8.2 or higher)
 * SciPy (version of 0.13.3 or higher)
 * Scikit-learn. Python 2.7 supports the versions of Scikit-learn 0.20 and earlier. In this study, the version of 0.20.2 was installed. Also, Numpy 1.15.4, 
 * Pandas 0.16.1
 * Matplotlib 1.4.3  
 In addition to these, the C++ compiler must be installed for windows.


### LSAT scripts have been prepared for the assessment of landslide susceptibility.  
 <p>
  <img width="1024" height="522" src="https://github.com/apolat2018/LSAT/blob/master/Figure1.jpg"><p>

 ## LSAT includes ten python script. These are:
* [Preparing_Data.py](https://github.com/apolat2018/LSAT/blob/master/Preparing_Data.py)
* [Create_LSM&Calculate_ROC.py](https://github.com/apolat2018/LSAT/blob/master/Create_LSM%26%26Calculate_ROC.py)
* [frequency_ratio.py](https://github.com/apolat2018/LSAT/blob/master/frequency_ratio.py)
* [information_value.py](https://github.com/apolat2018/LSAT/blob/master/information_value.py)
* [Logistic_Regression.py](https://github.com/apolat2018/LSAT/blob/master/Logistic_Regression.py)
* [tune_lr.py](https://github.com/apolat2018/LSAT/blob/master/tune_lr.py)
* [randomforest.py](https://github.com/apolat2018/LSAT/blob/master/randomforest.py)
* [tune_rf.py](https://github.com/apolat2018/LSAT/blob/master/tune_rf.py)
* [MLP.py](https://github.com/apolat2018/LSAT/blob/master/MLP.py)
* [tune_mlp.py](https://github.com/apolat2018/LSAT/blob/master/tune_mlp.py)

A tool file also was created for ArcGIS software (Landslide_Susceptibility_Assesment_Tool.tbx). 

 - Preparing_Data.py is used to prepare data for modeling as .csv format. 
 - Create_LSM&Calculate_ROC.py is used to creates Landslide Susceptibility Map and calculates Area Under Curve (AUC) values with data including X-Y coordinate and probability fields. Prepared data using this script can be analyzed in external software. Then classification results can be processed with Create LSM and Calculate ROC script in GIS and susceptibility map can be created with AUC values. 
 - The other scripts are used to create LSM with the methods of Frequency Ratio (FR), Information Value (IV), Logistic Regression (LR), Random Forest (RF) and Multi-Layer Perceptron (MLP). Also, this tool includes tuning script for the methods of LR, RF, and MLP.


**Supplementary**
|-----------|
|[Annex](https://github.com/apolat2018/LSAT/tree/master/Annex)|
[Figures](https://github.com/apolat2018/LSAT/tree/master/Figures)|

*Dr. Ali POLAT (2021)*

### Cite this article
Polat, A. An innovative, fast method for landslide susceptibility mapping using GIS-based LSAT toolbox. Environ Earth Sci 80, 217 (2021). https://doi.org/10.1007/s12665-021-09511-y
