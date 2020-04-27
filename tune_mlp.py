# -*- coding: cp1254 -*-
"""
This Script tunes the Multi Layer Perceptron Algorithm. RandomizedSearchCV method can be used.
A graph is plotted for every selecting parameter. Also the values of Success rate
and Prediction rate are seen on screen.

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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV
arcpy.env.overwriteOutput = True
#////////////////////////////Getting Input Parameters//////////////////////////
rec=arcpy.GetParameterAsText(0)#The folder including output data of Data Preparation script
sf=arcpy.GetParameterAsText(1)#output file is saving this Folder
wt=str(arcpy.GetParameterAsText(2))#weighting data type (Frequency ratio or Information Value)
tuning_parameter=arcpy.GetParameterAsText(3)#Select a parameter for tuning
h_layer=arcpy.GetParameterAsText(4)#hidden layer size
act=arcpy.GetParameterAsText(5)#activation
slv=arcpy.GetParameterAsText(6)#solver
alpha=float(arcpy.GetParameterAsText(7))#Alpha
l_rate=arcpy.GetParameterAsText(8)#learnning rate
l_rate_init=float(arcpy.GetParameterAsText(9))#learning rate init
max_it=int(arcpy.GetParameterAsText(10))#maximum iteration number
mom=float(arcpy.GetParameterAsText(11))#momentum
RS=arcpy.GetParameterAsText(12)
arcpy.AddMessage(RS)
arcpy.env.workspace=rec
#//////////////////checking Hidden layer size single or multi./////////////////
h_layer=h_layer.split(";")
layer_lst=[]

for h in h_layer:
    h=int(h)
    layer_lst.append(h)
arcpy.AddMessage(len(layer_lst))
if len(layer_lst)==1:
    hls=layer_lst[0]
else:
    hls=tuple(layer_lst)#tuple for Hidden layer size parameter
    
arcpy.AddMessage(layer_lst)

arcpy.AddMessage(hls)

os.chdir(rec)

arcpy.AddMessage("Starting MLP Analysis...")

#////////////////Starting Tuning///////////////////////////////////////////////
arcpy.AddMessage("Starting analysis with MLP algorithm")
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
#Loading train data
veriler=pd.read_csv(trn)
veriler=veriler.replace(-9999,"NaN")
#Loading analysis data
analiz=pd.read_csv(pre)
analiz=analiz.replace(-9999,"NaN")
#Loading validation dat
veriler_v=pd.read_csv(tst)
veriler_v=veriler_v.replace(-9999,"NaN")
#Preparing parameters
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
sc1=StandardScaler()
sc1.fit(parametreler)

parametreler=sc1.transform(parametreler)
pre=sc1.transform(pre)
param_validation=sc1.transform(param_validation)
#train-test splitting
pre=pd.DataFrame(data=pre)
x_train=pd.DataFrame(data=parametreler)
y_train=pd.DataFrame(data=cls)
x_test=pd.DataFrame(data=param_validation)
y_test=pd.DataFrame(data=cls_v)
#//////////////////////Tuning//////////////////////////////////////////////////
h_l_s = [5,10,15, 20, 30, 40, 50, 60, 70, 80,90,100]

activation=["identity", "logistic", "tanh", "relu"]
solvr=["lbfgs", "sgd", "adam"]
alph=[0.00001,0.0001,0.0005,0.001,0.01]
learning_r=["constant", "invscaling", "adaptive"]
learning_r_i=[0.0001,0.001,0.1,0.5]
max_iters=[250,500,1000,2000,5000]
momentums=[0.5,0.6,0.7,0.8,0.9]
train_results=[]
test_results=[]
os.chdir(sf)
#////////////////////////////////RandomizedSearchCV////////////////////////////
if RS=="true":
    arcpy.AddMessage("RandomizedsearchCV method was selected")
    arcpy.AddMessage("Please wait....This might take a while")
#.................................RandomizedSearchCV.................................
    model=MLPClassifier(tol=1e-5)
    parameters={"hidden_layer_sizes":[5,10,15, 20, 30, 40, 50, 60, 70, 80,90,100],"activation":["identity", "logistic", "tanh", "relu"],"solver":["lbfgs", "sgd", "adam"],
                "alpha":[0.00001,0.0001,0.0005,0.001,0.01],"learning_rate":["constant", "invscaling", "adaptive"],"learning_rate_init":[0.0001,0.001,0.1,0.5],
                "max_iter":[250,500,1000,2000,5000],"momentum":[0.5,0.6,0.7,0.8,0.9]}
    clf=RandomizedSearchCV(estimator=model,param_distributions=parameters,cv=5,random_state=0)
    clf.fit(x_train,y_train)
    arcpy.AddMessage("best parameteres={}".format(clf.best_params_))
#////////////////////////////////Other tuning proccesses///////////////////////
#Tuning graphs will be saved as .png file
else:

    if tuning_parameter=="hidden_layer_size":
        for i in h_l_s:
            arcpy.AddMessage("testing {} value".format(i))
            mlp=MLPClassifier(hidden_layer_sizes=i,activation=act,solver=slv,
                      alpha=alpha,learning_rate=l_rate,learning_rate_init=l_rate_init,max_iter=max_it,momentum=mom)
            mlp.fit(x_train, y_train)
            train_pred = mlp.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = mlp.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(h_l_s, train_results, "b", label="Train AUC")
        line2, = plt.plot(h_l_s, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    #    for a,b,c in zip(h_l_s,train_results,test_results):#show value of success and prediction
    #        plt.text(a,b,str(b))
    #        plt.text(a,c,str(c))
        plt.ylabel("AUC score")
        plt.xlabel("hidden_layer_size")
        plt.savefig("hidden_layer_size.png",dpi=150)
        plt.show()   
        plt.close("all")
    elif tuning_parameter=="activation_function":
        for i in activation:
            arcpy.AddMessage("testing {} value".format(i))
            mlp=MLPClassifier(hidden_layer_sizes=hls,activation=i,solver=slv,
                      alpha=alpha,learning_rate=l_rate,learning_rate_init=l_rate_init,max_iter=max_it,momentum=mom)
            mlp.fit(x_train, y_train)
            train_pred = mlp.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = mlp.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1,=plt.plot(range(len(activation)), train_results, "b", label="Train AUC")
        line2,=plt.plot(range(len(activation)), test_results, "r", label="Test AUC")
        plt.xticks(range(len(activation)),activation)
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("Activation function")
        plt.grid(True)
        plt.savefig("activation_f.png",dpi=150)
        plt.show()   
        plt.close("all")
    elif tuning_parameter=="solver":
        for i in solvr:
            arcpy.AddMessage("testing {} value".format(i))
            mlp=MLPClassifier(hidden_layer_sizes=hls,activation=act,solver=i,
                      alpha=alpha,learning_rate=l_rate,learning_rate_init=l_rate_init,max_iter=max_it,momentum=mom)
            mlp.fit(x_train, y_train)
            train_pred = mlp.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            y_pred = mlp.predict(x_test)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1,=plt.plot(range(len(solvr)), train_results, "b", label="Train AUC")
        line2,=plt.plot(range(len(solvr)), test_results, "r", label="Test AUC")
        
        plt.xticks(range(len(solvr)),solvr)
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("Solver")
        plt.grid(True)
        plt.savefig("solver.png",dpi=150)
        plt.show()   
        plt.close("all")
    elif tuning_parameter=="alpha":
        for i in alph:
            arcpy.AddMessage("testing {} value".format(i))
            mlp=MLPClassifier(hidden_layer_sizes=hls,activation=act,solver=slv,
                      alpha=i,learning_rate=l_rate,learning_rate_init=l_rate_init,max_iter=max_it,momentum=mom)
            mlp.fit(x_train, y_train)
            train_pred = mlp.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            y_pred = mlp.predict(x_test)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(alph, train_results, "b", label="Train AUC")
        line2, = plt.plot(alph, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("Alpha")
        plt.grid(True)
        plt.savefig("alpha.png",dpi=150)
        plt.show()   
        plt.close("all")
    elif tuning_parameter=="learning_rate":
        for i in learning_r:
            arcpy.AddMessage("testing {} value".format(i))
            mlp=MLPClassifier(hidden_layer_sizes=hls,activation=act,solver=slv,
                      alpha=alpha,learning_rate=i,learning_rate_init=l_rate_init,max_iter=max_it,momentum=mom)
            mlp.fit(x_train, y_train)
            train_pred = mlp.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = mlp.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1,=plt.plot(range(len(learning_r)), train_results, "b", label="Train AUC")
        line2,=plt.plot(range(len(learning_r)), test_results, "r", label="Test AUC")
        plt.xticks(range(len(learning_r)),learning_r)
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("Learning rate")
        plt.grid(True)
        plt.savefig("learning_rate.png",dpi=150)
        plt.show()   
        plt.close("all")
    elif tuning_parameter=="learning_rate_init":
        for i in learning_r_i:
            arcpy.AddMessage("testing {} value".format(i))
            mlp=MLPClassifier(hidden_layer_sizes=hls,activation=act,solver=slv,
                      alpha=alpha,learning_rate=l_rate,learning_rate_init=i,max_iter=max_it,momentum=mom)
            mlp.fit(x_train, y_train)
            train_pred = mlp.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            y_pred = mlp.predict(x_test)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(learning_r_i, train_results, "b", label="Train AUC")
        line2, = plt.plot(learning_r_i, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("learning_rate_init")
        plt.grid(True)
        plt.savefig("learning_rate_init.png",dpi=150)
        plt.show()   
        plt.close("all")
    elif tuning_parameter=="max_iter":
        for i in max_iters:
            arcpy.AddMessage("testing {} value".format(i))
            mlp=MLPClassifier(hidden_layer_sizes=hls,activation=act,solver=slv,
                      alpha=alpha,learning_rate=l_rate,learning_rate_init=l_rate_init,max_iter=i,momentum=mom)
            mlp.fit(x_train, y_train)
            train_pred = mlp.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = mlp.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(max_iters, train_results, "b", label="Train AUC")
        line2, = plt.plot(max_iters, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("Max_iter")
        plt.grid(True)
        plt.savefig("max_iter.png",dpi=150)
        plt.show()   
        plt.close("all")
    elif tuning_parameter=="momentum":
        for i in momentums:
            arcpy.AddMessage("testing {} value".format(i))
            mlp=MLPClassifier(hidden_layer_sizes=hls,activation=act,solver=slv,
                      alpha=alpha,learning_rate=l_rate,learning_rate_init=l_rate_init,max_iter=max_it,momentum=i)
            mlp.fit(x_train, y_train)
            train_pred = mlp.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            arcpy.AddMessage("Success rate={}".format(roc_auc))
            y_pred = mlp.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)
            arcpy.AddMessage("predict rate={}".format(roc_auc))
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(momentums, train_results, "b", label="Train AUC")
        line2, = plt.plot(momentums, test_results, "r", label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel("AUC score")
        plt.xlabel("Momentum")
        plt.grid(True)
        plt.savefig("momentum.png",dpi=150)
        plt.show()   
        plt.close("all")

arcpy.ClearWorkspaceCache_management()