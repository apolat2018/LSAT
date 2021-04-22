# *Landslide Susceptibility Assessment tool* (LSAT)

#### LSAT scripts have been prepared for the assessment of landslide susceptibility.  
 <p>
  <img width="1024" height="522" src="https://github.com/apolat2018/LSAT/blob/master/Figure1.jpg"><p>

 ### LSAT includes ten python script. These are:
* [Preparing_Data.py](https://github.com/apolat2018/LSAT/blob/master/Preparing_Data.py)
* [frequency_ratio.py](https://github.com/apolat2018/LSAT/blob/master/frequency_ratio.py)
* [information_value.py](https://github.com/apolat2018/LSAT/blob/master/information_value.py)
* [Logistic_Regression.py](https://github.com/apolat2018/LSAT/blob/master/Logistic_Regression.py)
* [randomforest.py](https://github.com/apolat2018/LSAT/blob/master/randomforest.py)
* [MLP.py](https://github.com/apolat2018/LSAT/blob/master/MLP.py)
* [tune_lr.py](https://github.com/apolat2018/LSAT/blob/master/tune_lr.py)
* [tune_rf.py](https://github.com/apolat2018/LSAT/blob/master/tune_rf.py)
* [tune_mlp.py](https://github.com/apolat2018/LSAT/blob/master/tune_mlp.py)
* [Create_LSM&Calculate_ROC.py](https://github.com/apolat2018/LSAT/blob/master/Create_LSM%26%26Calculate_ROC.py)

    * [LSAT tool file](https://github.com/apolat2018/LSAT/blob/master/Landslide_Susceptibility_Assesment_Tool.tbx). 

 - "Preparing Data.py" prepares data to be used in analysis.
 
 - The five scripts (frequency_ratio.py, information_value.py, Logistic_Regression.py, randomforest.py, and MLP.py) are used to create landslide susceptibility map with the methods of Frequency Ratio (FR), Information Value (IV), Logistic Regression (LR), Random Forest (RF), and Multi-Layer Perceptron (MLP).  
 - Also, this tool includes tuning scripts (tune_lr.py, tune_rf.py, tune_mlp.py) for the methods of LR, RF, and MLP.
 - Create_LSM&Calculate_ROC.py is used to creates Landslide Susceptibility Map and calculates Area Under Curve (AUC) values with data including X-Y coordinate and probability fields. Prepared data using this script can be analyzed in external software. Then classification results can be processed with Create LSM and Calculate ROC script in GIS and susceptibility map can be created with AUC values. 

## Preparing reclassed landslide factor data

Firstly you should prepare reclassed landslide factor raster files before using the toolbox. They should be the same sizes and same resolutions. Prepared raster files should be as shown in the figure below.
 <p>
  <img width="512" height="1024" src="https://github.com/apolat2018/LSAT/blob/master/rasters_image.jpg"><p>

IMPORTANT: All parameter files should be in the same folder. There should be no files other than factor raster files in this folder. The names of the parameters (factors) raster files must begin with "rec". rec_asp, rec_slp etc. Please do not use too long file names.

Landslide file must be polygonal type as shapefile format.\
Also, the area file must be polygonal type as shapefile format.\
The sample data folder is given below:\
https://github.com/apolat2018/LSAT/tree/master/sample_data \
Factor raster files are in raster folder. Landslides and area file are in vectors folder.
## Installing required libraries

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
## How to use the LSAT toolbox
* After the installations are done, download the toolbox and script files with py extensions to same folder.
* Open ArcGis.
* Go to Catalog and open the toolbox file in the downloaded folder.
* Please check file location for all scripts.To do this:
    * Right click on script
    * Select properties
    * Select search
    * choose script file
    * Do this for all script files in toolbox.
  <p>
  <img width="622" height="397" src=https://github.com/apolat2018/LSAT/blob/master/Annex/fig5.png><p>
### Preparing data for analysis
* Double click "1- Data Preparation" script to prepare data.
    * Select the folder of landslide parameters (The name of the parameter raster files must begin with "rec". rec_aspect, rec_slope etc.)
    * Select landslide shp file (Landslides file(.shp) must be polygon type)
    * Select area file (must be polygonal type .shp)
    * Select cell size
    * Select Train-Validation Split size (%). 70 mean %70 of data for train and 30% of data for test
    * Click "OK"

    <p>
  <img width="550" height="512" src="https://github.com/apolat2018/LSAT/blob/master/Annex/fig1.png"><p>

### Analysis
Choose an analysis method.\
For example, if you chose logistic regression:
* Double click "5- Logistic Regression method" 
* Select workspace folder (The folder including output data of Data Preparation script)
* Select save file folder (Susceptibility map and ROC graph is saving this Folder)
* Select Coordinate system
* Select cell size
* Select weighting data type frequency ratio or information value
* Select C value: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
* Select maximum iteration value:Maximum number of iterations taken for the solvers to converge.
* Select solver:'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'

This script will create landslide susceptiblity map with logistic regression method.\
You can choose the methods of Frequency Ratio (FR), Information Value (IV), Logistic Regression (LR), Random Forest (RF), and Multi-Layer Perceptron (MLP) with the LSAT tool.

If you have an excel file including x, y coordinate values, and probability fields you can use the script of "2- Create LSM and Calculate ROC". The excel file can be created by external software with different analysis methods. If you want to use the analysis methods of the LSAT tool you don't need to use this script.
<p>
  <img width="550" height="512" src=https://github.com/apolat2018/LSAT/blob/master/Annex/fig5.png><p>
 
**Supplementary**
|-----------|
|[Annex](https://github.com/apolat2018/LSAT/tree/master/Annex)|
[Figures](https://github.com/apolat2018/LSAT/tree/master/Figures)|

*Dr. Ali POLAT (2021)*\
ali.polat@afad.gov.tr

### Cite this article
Polat, A. An innovative, fast method for landslide susceptibility mapping using GIS-based LSAT toolbox. Environ Earth Sci 80, 217 (2021). https://doi.org/10.1007/s12665-021-09511-y
