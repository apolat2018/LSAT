# -*- coding: utf-8 -*-
"""
This Script tunes the Random Forest Algorithm. RandomizedSearchCV method can be used.
A graph is plotted for every selecting parameter. Also the values of Success rate and Prediction rate are seen on screen.

Created on Mon Nov  5 22:30:05 2018
@author: AP
"""
#////////////////////IMPORTING THE REQUIRED LIBRARIES/////////////////////////
import arcpy
import os
from arcpy.sa import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_curve, auc
arcpy.env.overwriteOutput = True
#////////////////////////////Getting Input Parameters//////////////////////////
rec=arcpy.GetParameterAsText(0)#The folder including output data of Data Preparation script
sf=arcpy.GetParameterAsText(1)#output file is saving this Folder
wt=arcpy.GetParameterAsText(2)#weighting data type (Frequency ratio or Information Value)
RS=arcpy.GetParameterAsText(3)#RandomizedSearchCV Method (use or not)
tuning_parameter=arcpy.GetParameterAsText(4)#Select a parameter for tuning
n_est=int(arcpy.GetParameterAsText(5))#The number of trees in the forest.
max_d=int(arcpy.GetParameterAsText(6))#The maximum depth of the tree.
min_samp_sp=str(arcpy.GetParameterAsText(7))#min_samples_split is a fraction 
#and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
min_samp_leaf=str(arcpy.GetParameterAsText(8))#The minimum number of samples required to be at a leaf node. 

arcpy.env.workspace=rec

arcpy.AddMessage(RS)

os.chdir(rec)
#///////////////////////Checking min_samp_sp Float or Integer//////////////////
for i in min_samp_sp:
    if i ==".":
        min_samp_sp=float(min_samp_sp)
    else:
        min_samp_sp=int(min_samp_sp)

for i in min_samp_leaf:
    if i ==".":
        min_samp_leaf=float(min_samp_leaf)
    else:
        min_samp_leaf=int(min_samp_leaf)

#//////Checking Weighting data Frequency ratio or İnformation value////////////
if wt=="frequency ratio":
    trn="train_fr.csv"
    pre="pre_fr.csv"
    tst="valid_fr.csv"
else:
    trn="train_iv.csv"
    pre="pre_iv.csv"
    tst="valid_iv.csv"
#Loading train data
veriler=pd.read_csv(trn)
veriler=veriler.replace(-9999,"NaN")
#Loading analysis data
analiz=pd.read_csv(pre)
analiz=analiz.replace(-9999,"NaN")
#Loading validation data
veriler_v=pd.read_csv(tst)
veriler_v=veriler_v.replace(-9999,"NaN")
####Preparing parameters
va,vb=veriler.shape
aa,ab=analiz.shape
ta,tb=veriler_v.shape
parametreler=veriler.iloc[:,2:vb].values
param_validation=veriler_v.iloc[:,2:tb].values
#Preparing label (class) data
cls=veriler.iloc[:,1:2].values
cls_v=veriler_v.iloc[:,1:2].values
#Preparing analysis data
pre=analiz.iloc[:,2:ab-2].values
##preparing Coordinate data 
koor=analiz.iloc[:,ab-2:ab].values

s_train=va
s_analiz=aa
koor=pd.DataFrame(data=koor,index=range(aa),columns=["x","y"])
#Converting NaN values to median
imputer= Imputer(missing_values='NaN', strategy = 'median', axis=0 )
parametreler=imputer.fit_transform(parametreler)
param_validation=imputer.fit_transform(param_validation)
pre=imputer.fit_transform(pre)
cls=imputer.fit_transform(cls)
cls_v=imputer.fit_transform(cls_v)
#train-test splitting
pre=pd.DataFrame(data=pre)
x_train=pd.DataFrame(data=parametreler)
y_train=pd.DataFrame(data=cls)
x_test=pd.DataFrame(data=param_validation)
y_test=pd.DataFrame(data=cls_v)

#//////////////////////Tuning//////////////////////////////////////////////////

p_n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
max_depths = np.linspace(1, 32, 32, endpoint=True)
list_md=list(max_depths)
p_min_samples_splits = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
p_min_samples_leaf=[0.1,0.12,0.14,0.16,0.18,0.2]
p_min_samples_splits1 =[2,3,4,5,6,7,8,9,10,15,20,25,50]
p_min_samples_leaf1=[10,50,100,250,500,1000]
os.chdir(sf)
train_results = []
test_results = []
#////////////////////////////////RandomizedSearchCV////////////////////////////
if RS =="true":
    arcpy.AddMessage("RandomizedsearchCV method was selected")
    arcpy.AddMessage("Please wait....This might take a while")
#.................................RandomizedSearchCV.................................
    model=RandomForestClassifier(random_state=0)
    parameters={"n_estimators":[1, 2, 4, 8, 16, 32, 64, 100, 200],"max_depth":list_md,"min_samples_split":p_min_samples_splits1,"criterion":["gini","entropy"],"min_samples_leaf":p_min_samples_leaf1}
    clf=RandomizedSearchCV(estimator=model,param_distributions=parameters,cv=5,random_state=0)
    clf.fit(x_train,y_train)
    arcpy.AddMessage("best parameteres={}".format(clf.best_params_))
#////////////////////////////////Other tuning proccesses///////////////////////
#Tuning graphs will be saved as .png file
else:
    
    if tuning_parameter=="n_estimators":
        arcpy.AddMessage("n_estimators was selected as tuning parameter")
         
        for i in p_n_estimators:
            arcpy.AddMessage("testing {} value".format(i))
            rf = RandomForestClassifier(n_estimators=i, max_depth=max_d,min_samples_split=min_samp_sp,min_samples_leaf=min_samp_leaf)
            rf.fit(x_train, y_train)
            train_pred = rf.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = rf.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(p_n_estimators, train_results, "b", label="Train AUC")
        line2, = plt.plot(p_n_estimators, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("n_estimators")
        plt.savefig("n_estimators.png",dpi=150)
        plt.show()   
        plt.close("all")
        arcpy.AddMessage("Graphic was saved as n_estimators.png ")
    elif tuning_parameter=="max_depth":
        arcpy.AddMessage("max_depth was selected as tuning parameter")
        for i in max_depths:
            arcpy.AddMessage("testing {} value".format(i))
            rf = RandomForestClassifier(n_estimators=n_est, max_depth=i,min_samples_split=min_samp_sp,min_samples_leaf=min_samp_leaf)
            rf.fit(x_train, y_train)
            train_pred = rf.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = rf.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(max_depths, train_results, "b", label="Train AUC")
        line2, = plt.plot(max_depths, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("max_depth")
        plt.savefig("max_depth.png",dpi=150)
        plt.show()   
        plt.close("all")
        arcpy.AddMessage("Graphic was saved as max_depth.png ")
    elif tuning_parameter=="min_samples_splits":
        arcpy.AddMessage("min_samples_splits was selected as tuning parameter") 
        for i in p_min_samples_splits1:
            arcpy.AddMessage("testing {} value".format(i))
            rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,min_samples_split=i,min_samples_leaf=min_samp_leaf)
            rf.fit(x_train, y_train)
            train_pred = rf.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = rf.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(p_min_samples_splits1, train_results, "b", label="Train AUC")
        line2, = plt.plot(p_min_samples_splits1, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("min_samples_splits")
        plt.savefig("min_samples_splits.png",dpi=150)
        plt.show()   
        plt.close("all")
        arcpy.AddMessage("Graphic was saved as min_sample_splits.png ")
    elif tuning_parameter=="min_samples_leaf":
        arcpy.AddMessage("min_samples_leaf was selected as tuning parameter") 
         
        for i in p_min_samples_leaf1:
            arcpy.AddMessage("testing {} value".format(i))
            rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,min_samples_split=min_samp_sp,min_samples_leaf=i)
            rf.fit(x_train, y_train)
            train_pred = rf.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = rf.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(p_min_samples_leaf1, train_results, "b", label="Train AUC")
        line2, = plt.plot(p_min_samples_leaf1, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("min_samples_leaf")
        plt.savefig("min_samples_leaf.png",dpi=150)
        plt.show()   
        plt.close("all")
        arcpy.AddMessage("Graphic was saved as min_samples_leaf.png ")
    
arcpy.ClearWorkspaceCache_management() 