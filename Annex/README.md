
<p align="center">
  <b>ANNEX</b>
</p>
<p><strong>User Interfaces and Processes of Landslide Susceptibility Tool Scripts</strong></p>
<p>Figures</p>

- Fig. 1 User interface of Data Preparation script
- Fig. 2 User interface of FR script
- Fig. 3 User interface of IV Script
- Fig. 4 User interface of Create LSM and Calculate ROC script.
- Fig. 5 User interface of LR script.
- Fig. 6 User interface of tuning LR script
- Fig. 7 User interface of RF script.
- Fig.  8 User interface of Tuning RF script.
- Fig.  9 User interface of MLP script.
- Fig. 10 User interface of tuning MLP script.

<p><strong>Tuning results of RF and MLP algorithms</strong></p>
<p>Figures</p>

- Fig. 8-1 Tuning results of RF algorithm
- Fig. 10-1 Tuning results of MLP algorithm

<p>Tables</p>

- Table 1 Description of input parameter and processes of Data Preparation Script. 
- Table 2 Description of input parameter and processes of FR Script.
- Table 3 Description of input parameter and processes of IV Script.
- Table 4 Description of input parameter and processes of Create LSM and Calculate ROC script.
- Table 5 Description of input parameter and processes of LR script.
- Table 6 Description of input parameter and processes of tuning LR script.
- Table 7 Description of input parameter and processes of RF script.
- Table 8 Description of input parameter and processes of Tuning RF script.
- Table 9 Description of input parameter and processes of MLP script.
- Table 10 Processes of tuning MLP script.

<p align="left">
  <img width="793" height="691" src="https://github.com/apolat2018/LSAT/blob/master/Annex/fig1.png">
</p>
<p align="left">
  <b>Figure 1.</b> User interface of Data Preparation script
</p>
<p align="left">
  <b>Table 1.</b> Description of input parameter and processes of Data Preparation Script. 
</p>

**DESCRIPTION OF INPUT PARAMETERS**
|Parameter|Explanation|Data Type|
|:-------- |:----------- |:----------|
|The_Folder_of_the_parameters|The name of the parameter, raster files must begin with "rec" rec_aspect, rec_slope etc.|Folder|
|Landslide_file__shp_	|Landslides file(.shp) must be polygon type|Shapefile|
|Area_file__shp_	|Desired area file (.shp)	|Shapefile|
|cell_size|	Cell size of the output raster data.|	Cell Size|
|Data_Type|	Two type of data is enabled for use in susceptibility analysis. These values are class values and normalized frequency ratio. 	|String|
|Train_Validation_Split_size|	Split size of data as train and validation. Percentage of input landslide data|	Double|

**THE PROCESSES OF DATA PREPARATION SCRIPT**
|:---------|
|[u'rec_asp', u'rec_crv', u'rec_eleva', u'rec_lito', u'rec_ls', u'rec_ndvi', u'rec_ridge', u'rec_slp', u'rec_tri', u'rec_twi']|
|10 raster data imported as parameter|
|Area is converting to Raster|
|Raster is converting to Point|

**PREPARING TRAIN DATA AND VALIDATION DATA**
|:---|
|Clipping landslides|
|landslide pixels are converting to point|
|Extracting landslide pixels|
|clp_asp is processing|
|Creating raster files with Normalized Frequency Ratios and Normalized Information Values|
|clp_asp is finished|
|Preparing Data for Frequency ratio value|
|Saving TRAINING data as train_fr.csv|
|Preparing Data for information value|
|Saving TRAINING data as train_iv.csv|
|Extracting Values to Point|
|Extracting finished|
|Saving ALL data as pre_iv.csv|
|Validation landslide is saving as test_1.shp|
|Completed script prepare...|
|*Elapsed Time: 14 minutes 15 seconds*|

<p align="left">
  <img width="793" height="691" src="https://github.com/apolat2018/LSAT/blob/master/Annex/fig2.png">
</p>
<p align="left">
  <b>Figure 2.</b> User interface of FR script
</p>

<p align="left">
  <b>Table 2.</b> Description of input parameter and processes of FR Script. 
</p>

**DESCRIPTION OF INPUT PARAMETER**

|Parameter|	Explanation|	Data Type|
|:---|:---|:---|
|Rec_folder|	Select the folder where the Parameter files (Reclassed raster data) are located.| 	Folder|
|Save_Folder|	Susceptibility map and ROC graph is saving this Folder|	Folder|
|Coordinate_system|	Coordinate System of output raster|	Coordinate System|
|cell_size|	Cell size of the raster|	Cell Size|
**THE PROCESSES OF FREQUENCY RATIO SCRIPT**
|Starting Frequency Ratio Analysis...|
|Analysis finished|
|Susceptibility Map was created in Save folder as FR_sus raster file|
|ROC calculation is starting.....|
|Success rate is: 72.7066604308|
|prediction rate is: 70.9536573075|
|AUC Graph is saved as auc_fr.png|
|Completed script FRanalysis...|
*(Elapsed Time: 26,56 seconds)*

<p align="left">
  <img width="793" height="691" src="https://github.com/apolat2018/LSAT/blob/master/Annex/fig3.png">
</p>
<p align="left">
  <b>Figure 3.</b> User interface of IV script
</p>

<p align="left">
  <b>Table 3.</b> Description of input parameter and processes of IV Script. 
</p>

**DESCRIPTION OF INPUT PARAMETER**

|Parameter|	Explanation|	Data Type|
|:---|:---|:---|
|Rec_folder|	Select the folder where the Parameter files (Reclassed raster data) are located.| 	Folder|
|Save_Folder|	Susceptibility map and ROC graph is saving this Folder|	Folder|
|Coordinate_system|	Coordinate System of output raster|	Coordinate System|
|cell_size|	Cell size of the raster|	Cell Size|
**THE PROCESSES OF FREQUENCY RATIO SCRIPT**
|Starting Information Value Analysis...|
|Analysis finished|
|Susceptibility Map was created in Save folder as IV_sus raster file|
|ROC calculation is starting.....|
|Success rate is: 72.942322706|
|prediction rate is: 71.8463442428|
|AUC Graph is saved as auc_IV.png|
|Completed script IVanalysis...|
*(Elapsed Time: 15,67 seconds)*

<p align="left">
  <img width="793" height="691" src="https://github.com/apolat2018/LSAT/blob/master/Annex/fig4.png">
</p>
<p align="left">
  <b>Figure 4.</b> User interface of Create LSM and Calculate ROC script.
</p>

<p align="left">
  <b>Table 4.</b> Description of input parameter and processes of Create LSM and Calculate ROC script.
</p>

**DESCRIPTION OF INPUT PARAMETER**

|Parameter|	Explanation|	Data Type|
|:---|:---|:---|
|workspace|	The folder including exported files|	Folder|
|excel_file|	excel file including x, y coordinate values and probability fields. It can be created external software or Data preparation script|	File|
|train|	landslide train data (.shp). if Data Preparation script was used you should output of this script. It is located in The Folder of Parameters.|	Shapefile|
|test|	landslide test (validation) data (.shp). if Data Preparation script was used you should output of this script. It is in The Folder of Parameters.|	Shapefile|
|Coordinate_System|	Select Coordinate System of raster files|	Coordinate System|
|Raster_map_Name|	Susceptibility map name|	String|
|cell_size|	Cell size of rasters|	Cell Size|
|field|	probability field name. The column name including probability values. Defaults is "ones".|Field|
**THE PROCESSES OF FREQUENCY RATIO SCRIPT**
|Running script Crate...|
|logr_gis.xlsx file is imported|
|Susceptibility map is saved as weka_lr|
|ROC is calculating|
|Success rate is: 75.4174228712|
|prediction rate is: 72.5975597239|
|AUC Graph is saved as auc_weka.png|
|FINISHED|
|Completed script Ceate...|
*Time: 2 minutes 7 seconds*

<p align="left">
  <img width="793" height="691" src="https://github.com/apolat2018/LSAT/blob/master/Annex/fig5.png">
</p>
<p align="left">
  <b>Figure 5.</b> User interface of LR script.
</p>

<p align="left">
  <b>Table 5.</b> Description of input parameter and processes of LR script.
</p>

**DESCRIPTION OF INPUT PARAMETER**

|Parameter|	Explanation|	Data Type|
|:---|:---|:---|
|workspace|	The folder including output data of Data Preparation script|	Folder|
|Save_file|	Susceptibility map and ROC graph is saving this Folder|	Folder|
|Coordinat_system|	Coordinate system|	Coordinate System|
|cell_size|	Cell size of the raster|	Long|
|Select_weighting_data_type|	Select weighting data type frequency ratio or information value	|String|
|C|	Regularization parameter|	Double|
|max_iter|	Maximum number of iterations| 	Long|
|solver|	'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'|	String|
**THE PROCESSES OF LR SCRIPT**
|Running script Logr...|
|Starting analysis with Logistic Regression algorithm|
|Data type= information value|
|C value:----------------:50.0|
|max_iter:---------------:100|
|Solver:-----------------:lbfgs|
|Starting Training|
|Saving Prediction Data as log_r.csv|
|Analysis finished|
|Creating SUSCEPTIBILITY Map and Calculating ROC|
|Susceptibility Map was created in save folder as logr_sus raster file|
|Success rate is: 75.400834642|
|prediction rate is: 72.5659791817|
|AUC Graph is saved as auc_logr.png|
|FINISHED|
|Completed script Logr...|
*Elapsed Time: 58,76 seconds*

<p align="left">
  <img width="793" height="691" src="https://github.com/apolat2018/LSAT/blob/master/Annex/fig6.png">
</p>
<p align="left">
  <b>Figure 6.</b> User interface of tuning LR script
</p>

<p align="left">
  <b>Table 6.</b> Description of input parameter and processes of tuning LR script.
</p>

**THE PROCESSES OF TUNING LR SCRIPT*
|----------------------------------|
|Selected C value as tuning parameter|
|testing 0.001 value|
|Success rate=0.664567830567|
|Predict rate=0.663840873377|
|testing 0.002 value|
|Success rate=0.669814920817|
|Predict rate=0.67041601191|
|testing 0.003 value|
|Success rate=0.6732732303|
|Predict rate=0.67401372922|
|testing 0.004 value|
|Success rate=0.674871207785|
|Predict rate=0.677073856588|
|testing 0.005 value|
|Success rate=0.676206830757|
|Predict rate=0.679141510214|
|testing 0.006 value|
|Success rate=0.677709406602|
|Predict rate=0.67988586552|
|testing 0.007 value|
|Success rate=0.678901927113|
|Predict rate=0.681539988421|
|testing 0.008 value|
|Success rate=0.680118298035|
|Predict rate=0.68265652138|
|testing 0.009 value|
|Success rate=0.680523755009|
|Predict rate=0.682739227525|
|testing 0.01 value|
|Success rate=0.680929211982|
|Predict rate=0.683235464395|
|Completed script tuningLR...|
*Elapsed Time: 12.68 seconds*

<p align="left">
  <img width="793" height="691" src="https://github.com/apolat2018/LSAT/blob/master/Annex/fig7.png">
</p>
<p align="left">
  <b>Figure 7.</b> User interface of RF script
</p>

<p align="left">
  <b>Table 7.</b> Description of input parameter and processes of RF script.
</p>

**DESCRIPTION OF INPUT PARAMETER**

|Parameter|	Explanation|	Data Type|
|:---|:---|:---|
|workspace|	The folder including output data of Data Preparation script|	Folder|
|Save_file|	Susceptibility map and ROC graph is saving this Folder|	Folder|
|Coordinat_system|	coordinate system of output map|	Coordinate System|
|cell_size|	cell size of output map|	Long|
|Select_weighting_data_type|	Select weighting data type frequency ratio or information value	|String|
|n_estimators|	The number of trees in the forest.|	Long|
|max_depth|	The maximum depth of the tree.|	Long|
|min_samples_split|	min_samples_split| 	Double|
|min_samples_leaf|	min_samples_leaf|	Double|
**THE PROCESSES OF RANDOM FOREST SCRIPT**
|Running script RF...|
|Starting analysis with rf algorithm|
|n_estimator:-----------:2000|
|Max depth:-------------:4|
|Min_sample_size:-------:2|
|Min_sample_leaf:-------:100|
|Starting Training.|
|Saving Prediction Data as rf.csv|
|Analysis finished|
|Creating SUSCEPTIBILITY Map and Calculating ROC |
|Susceptibility Map was created in rslts folder as rf_sus raster file|
|Success rate is: 74.4710490374|
|prediction rate is: 71.2699643053|
|AUC Graph is saved as auc_rf.png|
|FINISHED|
|Completed script RF...|
*Elapsed Time: 53,47 seconds*

<p align="left">
  <img width="793" height="691" src="https://github.com/apolat2018/LSAT/blob/master/Annex/fig8.png">
</p>
<p align="left">
  <b>Figure 8.</b> User interface of Tuning RF script.
</p>

<p align="left">
  <b>Table 8.</b> Description of input parameter and processes of Tuning RF script.
</p>

**THE PROCESSES OF TUNING RANDOM FOREST SCRIPT**
|---------------------------------------------|
|n_estimators was selected as tuning parameter|
|testing 1 value|
|Success rate=0.712125548559|
|predict rate=0.642709453312|
|testing 2 value|
|Success rate=0.735069643198|
|predict rate=0.694897030849|
|testing 4 value|
|Success rate=0.732732302996|
|predict rate=0.683566288975|
|testing 8 value|
|Success rate=0.757632131273|
|predict rate=0.687412124721|
|testing 16 value|
|Success rate=0.750095401641|
|predict rate=0.701802993962|
|testing 32 value|
|Success rate=0.755986452967|
|predict rate=0.697502274419|
|testing 64 value|
|Success rate=0.752957450868|
|predict rate=0.694731618559|
|testing 100 value|
|Success rate=0.756511161992|
|predict rate=0.697667686709|
|testing 200 value|
|Success rate=0.753744514406|
|predict rate=0.698412042015|
|Graphic was saved as n_estimators.png|
|Completed script RFTune...|
*Elapsed Time: 19,80 seconds*

<p align="left">
  <img width="793" height="691" src="https://github.com/apolat2018/LSAT/blob/master/Annex/fig8-1.jpg">
</p>
<p align="left">

<p align="left">
  <b>Figure 8-1.</b> Tuning results of RF algorithm: (a)n-estimators, (b) minimum sample splits, (c) maximum depth, (d) minimum sample leaf.
</p>

<p align="left">
  <img width="793" height="691" src="https://github.com/apolat2018/LSAT/blob/master/Annex/fig9.png">
</p>
<p align="left">
  <b>Figure 9.</b> User interface of MLP script.
</p>

<p align="left">
  <b>Table 9.</b> Description of input parameter and processes of MLP script.
</p>

**DESCRIPTION OF INPUT PARAMETER**

|Parameter|	Explanation|	Data Type|
|:---|:---|:---|
|workspace|	The folder including output data of Data Preparation| script	Folder|
|Save_file|	Susceptibility map and ROC graph is saving this Folder|	Folder|
|Coordinat_system|	coordinate system of output map|	Coordinate System|
|cell_size|	cell size of output map|	Long|
|Select_weighting_data_type|	Select weighting data type frequency ratio or information value|	String|
|hidden_layer_size|	Hidden layer size|	Multiple Value|
|activation|	Activation function|	String|
|solver|	Solver|	String|
|alpha|	Regularization parameter.|	Double|
|learning_rate|	Learning rate|	String|
|learning_rate_init|	Learning rate init|	Double|
|max_iter|	Max iteration number|	Long|
|momentum|	Momentum|	Double|
**THE PROCCESES OF MLP SCRIPT**
|Running script MLP...|
|Hidden layer size:---------------:100|
|Activation function:-------------:tanh|
|Solver:--------------------------:sgd|
|Alpha----------------------------:1e-05|
|Learning rate:-------------------:constant|
|Learning rate init---------------:0.0001|
|Max iter:------------------------:2000|
|Momentum-------------------------:0.7|
|Starting analysis with MLP algorithm|
|Starting Training|
|Saving Prediction Data as mlp.csv|
|Analysis finished|
|Creating SUSCEPTIBILITY Map and Calculating ROC| 
|Susceptibility Map was created in C:\deneme\rslts folder as mlp_sus raster file|
|Success rate is: 75.3942743596|
|prediction rate is: 72.6741995105|
|AUC Graph is saved as auc_mlp.png|
|Completed script MLP...|
*Elapsed Time: 10 minutes 24seconds*

<p align="left">
  <img width="793" height="691" src="https://github.com/apolat2018/LSAT/blob/master/Annex/fig10.png">
</p>
<p align="left">
  <b>Figure 10.</b> User interface of tuning MLP script.
</p>

<p align="left">
  <b>Table 10.</b> Processes of tuning MLP script.
</p>

**THE PROCESSES OF TUNING MLP SCRIPT**
|-------------------------------------|
|Selected hidden layer size as tuning parameter|
|Testing values of 5,10,15, 20, 30, 40, 50, 60, 70, 80,90,100|
|testing 5 value|
|Success rate=0.726435794696|
|predict rate=0.697378215201|
|testing 10 value|
|Success rate=0.73106277428|
|predict rate=0.698453395087|
|testing 15 value|
|Success rate=0.745253768365|
|predict rate=0.701182697874|
|testing 20 value|
|Success rate=0.744991413852|
|predict rate=0.701141344802|
|testing 30 value|
|Success rate=0.752241938561|
|predict rate=0.693491026383|
|testing 40 value|
|Success rate=0.749499141385|
|predict rate=0.701017285584|
|testing 50 value|
|Success rate=0.752194237741|
|predict rate=0.69001736829|
|testing 60 value|
|Success rate=0.747448006106|
|predict rate=0.69539326772|
|testing 70 value|
|Success rate=0.753458309483|
|predict rate=0.697915805144|
|testing 80 value|
|Success rate=0.755557145583|
|predict rate=0.69866016045|
|testing 90 value|
|Success rate=0.753315207022|
|predict rate=0.702216524688|
|testing 100 value|
|Success rate=0.754436176302|
|predict rate=0.692333140352|
|Completed script MLP2...|
*Elapsed Time: 5 minutes 42 seconds*

<p align="left">
  <img width="793" height="691" src="https://github.com/apolat2018/LSAT/blob/master/Annex/fig10-1a.jpg">
  <img width="793" height="691" src="https://github.com/apolat2018/LSAT/blob/master/Annex/fig10-1b.jpg">
</p>
<p align="left">

<p align="left">
  <b>Figure 10-1.</b> Tuning parameters of MLP algorithm. (a) activation functions, (b) alpha, (c) hidden layer size, (d) learning rate, (e) learning rate init, (f) maximum iteration, (g) momentum, (h) solver functions
</p>

[go to master](https://github.com/apolat2018/LSAT/tree/master)
[Figures](https://github.com/apolat2018/LSAT/tree/master/Figures)
