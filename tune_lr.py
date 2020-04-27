# -*- coding: utf-8 -*-
"""
This Script tunes the Logistic Regression Algorithm. GridsearcCV method can be used.
A graph is plotted for every selecting parameter. Also the values of Success rate and Prediction rate are seen on screen.

Created on Mon Nov  5 22:30:05 2018
@author: Ali POLAT
"""
#////////////////////IMPORTING THE REQUIRED LIBRARIES//////////////////////////
import arcpy
import os
from arcpy.sa import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_curve, auc
arcpy.env.overwriteOutput = True
#////////////////////////////Getting Input Parameters//////////////////////////
rec=arcpy.GetParameterAsText(0)#The folder including output data of Data Preparation script
sf=arcpy.GetParameterAsText(1)#output file is saving this Folder
wt=str(arcpy.GetParameterAsText(2))#weighting data type (Frequency ratio or Information Value)
GS=arcpy.GetParameterAsText(3)#GridseaerchCV Method (use or not)
tps=str(arcpy.GetParameterAsText(4))#Select a parameter for tuning. 
single_c=float(arcpy.GetParameterAsText(5))#Use a unique float number tuning for max_iter or solver. 
selection=arcpy.GetParameterAsText(6)#select c value range
mi=int(arcpy.GetParameterAsText(7))#max_iter
slvr=str(arcpy.GetParameterAsText(8))#solver
#///////////////////////////Assigning selections///////////////////////////////
selection=str(selection)
sel=selection.split("=")
sel=sel[0]
##arcpy.AddMessage(rec_list)
os.chdir(rec)
#//////Checking Weighting data Frequency ratio or İnformation value////////////
if wt=="frequency ratio":
    trn="train_fr.csv"
    pre="pre_fr.csv"
    tst="valid_fr.csv"
else:
    trn="train_iv.csv"
    pre="pre_iv.csv"
    tst="valid_iv.csv"
#Loading Train Data 
veriler=pd.read_csv(trn)
veriler=veriler.replace(-9999,"NaN")
#Loading Analysis data
analiz=pd.read_csv(pre)
analiz=analiz.replace(-9999,"NaN")
#Loading Validation data
veriler_v=pd.read_csv(tst)
veriler_v=veriler_v.replace(-9999,"NaN")
##Preparing parameters
va,vb=veriler.shape
aa,ab=analiz.shape
ta,tb=veriler_v.shape
parametreler=veriler.iloc[:,2:vb].values
param_validation=veriler_v.iloc[:,2:tb].values
#Preparing label (class) data
cls=veriler.iloc[:,1:2].values
cls_v=veriler_v.iloc[:,1:2].values
#preparing Analysis data
pre=analiz.iloc[:,2:ab-2].values
#preparing Coordinate data 
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
c_range1 = np.linspace(0.0001,0.001,10,endpoint=True)
c_range2 = np.linspace(0.001,0.01,10,endpoint=True)
c_range3 = np.linspace(0.01,0.1,10,endpoint=True)
c_range4 = np.linspace(0.1,1,10,endpoint=True)
c_range5 = [10,50,100,250,500,750,1000]
max_iters=[10,100,200,300,500,1000]
solvers=["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
os.chdir(sf)
ranges=[c_range1,c_range2,c_range3,c_range4,c_range5]
train_results = []
test_results = []
#////////////////////////////////GridSearchCV//////////////////////////////////
if GS=="true":
    arcpy.AddMessage("Gridsearch method was selected")
    arcpy.AddMessage("Please wait....This might take a while")
    model=LogisticRegression(random_state=0)
    parameters={"C":[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,10,50,200,500],"max_iter":[10,100,200,300,500],"solver":("newton-cg", "lbfgs", "liblinear", "sag", "saga")}
    clf=GridSearchCV(model,parameters,cv=5)
    clf.fit(x_train,y_train)
    arcpy.AddMessage("best parameteres={}".format(clf.best_params_))
#////////////////////////////////Other tuning proccesses///////////////////////
#Tuning graphs will be saved as .png file
else:
    if tps=="c_value" and sel=="range1":
        for i in c_range1:
            arcpy.AddMessage("testing {} value".format(i))
            lr = LogisticRegression(C=i,random_state=0,solver=slvr,max_iter=mi)
            lr.fit(x_train, y_train)
            train_pred = lr.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = lr.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(c_range1, train_results, "b", label="Train AUC")
        line2, = plt.plot(c_range1, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("C values")
        plt.savefig("c_values.png",dpi=150)
        plt.show()   
        plt.close("all")
    elif tps=="c_value" and sel=="range2":
        for i in c_range2:
            arcpy.AddMessage("testing {} value".format(i))
            lr = LogisticRegression(C=i,random_state=0,solver=slvr,max_iter=mi)
            lr.fit(x_train, y_train)
            train_pred = lr.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            y_pred = lr.predict(x_test)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("Predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(c_range2, train_results, "b", label="Train AUC")
        line2, = plt.plot(c_range2, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("C values")
        plt.savefig("c_values.png",dpi=150)
        plt.show()   
        plt.close("all")
    elif tps=="c_value" and sel=="range3":
        for i in c_range3:
            arcpy.AddMessage("testing {} value".format(i))
            lr = LogisticRegression(C=i,random_state=0,solver=slvr,max_iter=mi)
            lr.fit(x_train, y_train)
            train_pred = lr.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = lr.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("Predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(c_range3, train_results, "b", label="Train AUC")
        line2, = plt.plot(c_range3, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("C values")
        plt.savefig("c_values.png",dpi=150)
        plt.show()   
        plt.close("all")
    elif tps=="c_value" and sel=="range4":
        for i in c_range4:
            arcpy.AddMessage("testing {} value".format(i))
            lr = LogisticRegression(C=i,random_state=0,solver=slvr,max_iter=mi)
            lr.fit(x_train, y_train)
            train_pred = lr.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = lr.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("Predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(c_range4, train_results, "b", label="Train AUC")
        line2, = plt.plot(c_range4, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("C values")
        plt.savefig("c_values.png",dpi=150)
        plt.show()   
        plt.close("all")
    elif tps=="c_value" and sel=="range5":
        for i in c_range5:
            arcpy.AddMessage("testing {} value".format(i))
            lr = LogisticRegression(C=i,random_state=0,solver=slvr,max_iter=mi)
            lr.fit(x_train, y_train)
            train_pred = lr.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = lr.predict(x_test)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("Predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(c_range5, train_results, "b", label="Train AUC")
        line2, = plt.plot(c_range5, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("C values")
        plt.savefig("c_values.png",dpi=150)
        plt.show()   
        plt.close("all")
    elif tps=="c_value" and sel=="range5":
        for i in c_range5:
            arcpy.AddMessage("testing {} value".format(i))
            lr = LogisticRegression(C=i,random_state=0,solver=slvr,max_iter=mi)
            lr.fit(x_train, y_train)
            train_pred = lr.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = lr.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("Predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(c_range5, train_results, "b", label="Train AUC")
        line2, = plt.plot(c_range5, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("C values")
        plt.savefig("c_values.png",dpi=150)
        plt.show()   
        plt.close("all")
    elif tps=="max_iter":
        for i in max_iters:
            arcpy.AddMessage("testing {} value".format(i))
            lr = LogisticRegression(C=single_c,random_state=0,solver=slvr,max_iter=i)
            lr.fit(x_train, y_train)
            train_pred = lr.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = lr.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("Predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(max_iters, train_results, "b", label="Train AUC")
        line2, = plt.plot(max_iters, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("Max_iter")
        plt.savefig("lr_max_iter.png",dpi=150)
        plt.show()   
        plt.close("all")
    elif tps=="solver":
        for i in solvers:
            arcpy.AddMessage("testing {} value".format(i))
            lr = LogisticRegression(C=single_c,random_state=0,solver=i,max_iter=mi)
            lr.fit(x_train, y_train)
            train_pred = lr.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = lr.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("Predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(range(len(solvers)), train_results, "b", label="Train AUC")
        line2, = plt.plot(range(len(solvers)), test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.xticks(range(len(solvers)),solvers)
        plt.ylabel("AUC score")
        plt.xlabel("Solver")
        plt.savefig("lr_solver.png",dpi=150)
        plt.show()   
        plt.close("all")
arcpy.ClearWorkspaceCache_management() 