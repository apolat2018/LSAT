# *Landslide Susceptibility Assessment tool* (LSAT)

 <p>LSAT scripts have been prepared for the assessment of landslide susceptibility.  
  <p align="left">
  <img width="1024" height="522" src="https://github.com/apolat2018/LSAT/blob/master/Figure1.jpg">
</p>
 LSAT includes ten python script. These are:
- [Preparing_Data.py](https://github.com/apolat2018/LSAT/blob/master/Preparing_Data.py)
- [Create_LSM&Calculate_ROC.py](https://github.com/apolat2018/LSAT/blob/master/Create_LSM%26%26Calculate_ROC.py)
- [frequency_ratio.py](https://github.com/apolat2018/LSAT/blob/master/frequency_ratio.py)
- [information_value.py](https://github.com/apolat2018/LSAT/blob/master/information_value.py)
- [Logistic_Regression.py](https://github.com/apolat2018/LSAT/blob/master/Logistic_Regression.py)
- [tune_lr.py](https://github.com/apolat2018/LSAT/blob/master/tune_lr.py)
- [randomforest.py](https://github.com/apolat2018/LSAT/blob/master/randomforest.py)
- [tune_rf.py](https://github.com/apolat2018/LSAT/blob/master/tune_rf.py)
- [MLP.py](https://github.com/apolat2018/LSAT/blob/master/MLP.py)
- [tune_mlp.py](https://github.com/apolat2018/LSAT/blob/master/tune_mlp.py)

A tool file also was created for ArcGIS software (Landslide_Susceptibility_Assesment_Tool.tbx). 

 - Preparing_Data.py is used to prepare data for modeling as .csv format. 
 - Create_LSM&Calculate_ROC.py is used to creates Landslide Susceptibility Map and calculates Area Under Curve (AUC) values with data including X-Y coordinate and probability fields. Prepared data using this script can be analyzed in external software. Then classification results can be processed with Create LSM and Calculate ROC script in GIS and susceptibility map can be created with AUC values. 
 - The other scripts are used to create LSM with the methods of Frequency Ratio (FR), Information Value (IV), Logistic Regression (LR), Random Forest (RF) and Multi-Layer Perceptron (MLP). Also, this tool includes tuning script for the methods of LR, RF, and MLP.
</p>

**Supplementary**
|-----------|
|[Annex](https://github.com/apolat2018/LSAT/tree/master/Annex)|
[Figures](https://github.com/apolat2018/LSAT/tree/master/Figures)|

*Dr. Ali POLAT (2020)*
