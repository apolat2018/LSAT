# -*- coding: cp1254 -*-
""" 
This script creates Landslide Susceptibility Map (LSM) with Multi Layer Perceptron Model 

Ali POLAT (2018)
"""
#////////////////////IMPORTING THE REQUIRED LIBRARIES/////////////////////////
import arcpy
import os
from arcpy.sa import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import Imputer 
from sklearn.preprocessing import StandardScaler
arcpy.env.overwriteOutput = True
#////////////////////////////Getting Input Parameters//////////////////////////
rec=arcpy.GetParameterAsText(0)#The folder including output data of Data Preparation script
sf=arcpy.GetParameterAsText(1)#output file is saving this Folder
koordinat=arcpy.GetParameterAsText(2)#Coordinate system of map
cell_size=arcpy.GetParameterAsText(3)#Cell size
wt=str(arcpy.GetParameterAsText(4))#weighting data type (Frequency ratio or Information value)
h_layer=arcpy.GetParameterAsText(5)#hidden layer size
act=arcpy.GetParameterAsText(6)#activation
slv=arcpy.GetParameterAsText(7)#solver
alpha=float(arcpy.GetParameterAsText(8))#Alpha
l_rate=arcpy.GetParameterAsText(9)#learnning rate
l_rate_init=float(arcpy.GetParameterAsText(10))#learning rate init
max_it=int(arcpy.GetParameterAsText(11))#maximum number of iteration
mom=float(arcpy.GetParameterAsText(12))#momentum

arcpy.env.workspace=rec
#//////////////////checking Hidden layer size single or multi./////////////////
h_layer=h_layer.split(";")
layer_lst=[]

for h in h_layer:
    h=int(h)
    layer_lst.append(h)

if len(layer_lst)==1:
    hls=layer_lst[0]
else:
    hls=tuple(layer_lst)#tuple for Hidden layer size parameter
#/////////showing parameter on screen//////////////////////////////////////////
arcpy.AddMessage("Hidden layer size:---------------:{}".format(hls))
arcpy.AddMessage("Activation function:-------------:{}".format(act))
arcpy.AddMessage("Solver:--------------------------:{}".format(slv))
arcpy.AddMessage("Alpha----------------------------:{}".format(alpha))
arcpy.AddMessage("Learning rate:-------------------:{}".format(l_rate))
arcpy.AddMessage("Learning rate init---------------:{}".format(l_rate_init))
arcpy.AddMessage("Max iter:------------------------:{}".format(max_it))
arcpy.AddMessage("Momentum-------------------------:{}".format(mom))   

os.chdir(rec)

#///////////////////////////Starting Anlysis///////////////////////////////////
arcpy.AddMessage("Starting analysis with MLP algorithm")
os.chdir(rec)
#//////Checking Weighting data Frequency ratio or İnformation value////////////
if wt=="frequency ratio":
    trn="train_fr.csv"
    pre="pre_fr.csv"
    
else:
    trn="train_iv.csv"
    pre="pre_iv.csv"
#Loading Train data
veriler=pd.read_csv(trn)
veriler=veriler.replace(-9999,"NaN")
#Loading Analysis data
analiz=pd.read_csv(pre)
analiz=analiz.replace(-9999,"NaN")
#Preparing parameters
va,vb=veriler.shape
aa,ab=analiz.shape
parametreler=veriler.iloc[:,2:vb].values
#Preparing label (class) data
cls=veriler.iloc[:,1:2].values
##preparing Analysis data
pre=analiz.iloc[:,2:ab-2].values
#preparing Coordinate data
koor=analiz.iloc[:,ab-2:ab].values

s_train=va
s_analiz=aa
koor=pd.DataFrame(data=koor,index=range(aa),columns=["x","y"])
#Converting NaN values to median
imputer= Imputer(missing_values='NaN', strategy = 'median', axis=0 )
parametreler=imputer.fit_transform(parametreler)
pre=imputer.fit_transform(pre)
cls=imputer.fit_transform(cls)
sc1=StandardScaler()
sc1.fit(parametreler)

parametreler=sc1.transform(parametreler)
pre=sc1.transform(pre)
pre=pd.DataFrame(data=pre)
x=pd.DataFrame(data=parametreler)
y=pd.DataFrame(data=cls)
#///////////////////////////Creating Model/////////////////////////////////////
mlp=MLPClassifier(hidden_layer_sizes=hls,activation=act,solver=slv,learning_rate=l_rate,learning_rate_init=l_rate_init,
                  alpha=alpha,max_iter=max_it,random_state=0,tol=1e-5,momentum=mom)
#//////////////////////////////////training model////////////////////////////// 
mlp.fit(x,y.values.ravel())
arcpy.AddMessage("Starting Training")
#//////////////////////////////////making prediction///////////////////////////
tahmin=mlp.predict_proba(pre)
mlp_cls=pd.DataFrame(data=tahmin,index=range(s_analiz),columns=["zeros","ones"])
K=pd.concat([koor,mlp_cls],axis=1)
arcpy.AddMessage("Saving Prediction Data as mlp.csv")
mlp_result=os.path.join(sf,"mlp.csv")
K.to_csv(mlp_result,columns=["x","y","ones"])
#///////////////////////////////Saving Prediction Data as mlp.csv////////////
arcpy.AddMessage("mlp best loss value: {}".format(mlp.best_loss_))# if desired MLP best loss value can be obtained
#//////////////////////////Creating Susceptibility map/////////////////////////
arcpy.AddMessage("Analysis finished")
mlp_sus_map=os.path.join(sf,"mlp_sus")
arcpy.AddMessage("Creating SUSCEPTIBILITY Map and Calculating ROC ")          
arcpy.MakeXYEventLayer_management(mlp_result,"x","y","model",koordinat,"ones")
arcpy.PointToRaster_conversion("model","ones",mlp_sus_map,"MOST_FREQUENT","",cell_size)
arcpy.AddMessage("Susceptibility Map was created in {} folder as mlp_sus raster file".format(sf))

#////////////////////////////CALCULATING PERFORMANCE///////////////////////////

mx=float (arcpy.GetRasterProperties_management (mlp_sus_map, "MAXIMUM").getOutput (0))
mn=float (arcpy.GetRasterProperties_management (mlp_sus_map, "MINIMUM").getOutput (0))

e=(float(mx)-float(mn))/100

d=[]
x=0
y=0
z=0
for f in range (100):
    
    x=x+1
    y=mn+e
    z=z+mn
    q=[]
    q.append(z)
    q.append(y)
    q.append(x)
    d.append(q)
    mn=y
    z=0

total=Reclassify(mlp_sus_map,"VALUE",RemapRange(d),"NODATA")
total_exp="total.tif"
total.save(total_exp)

trn=ExtractByMask(total,"train_1.shp")
train_exp="train.tif"
trn.save(train_exp)

tes=ExtractByMask(total,"test_1.shp")
test_exp="test.tif"
tes.save(test_exp)
##.............................................
arcpy.AddField_management(total_exp,"total","DOUBLE")
arcpy.AddField_management(total_exp,"NID","LONG")

block="""rec=0
def yaz():
    global rec
    pstart=1
    pinterval=1
    if(rec==0):
        rec=pstart
    else:
        rec+=pinterval
    return rec"""
expression="yaz()"
arcpy.CalculateField_management(total_exp,"NID",expression,"PYTHON",block)
lst_nid=list()

with arcpy.da.SearchCursor(total_exp,"NID") as dd:
    for row in dd:
        lst_nid.append(row[0])
        
        del row
    
    del dd

mx=max(lst_nid)
crs=arcpy.da.InsertCursor(total_exp,["NID"])
for i in range(mx+1,101):
   crs.insertRow("0")
arcpy.CalculateField_management(total_exp,"NID",expression,"PYTHON",block)
lst_value=[]
lst_count=[]
lst_nc=[]
lst_nid_2=[]
sc_fields="value","count","total","NID"
with arcpy.da.SearchCursor(total_exp,sc_fields) as scur:
    for row in scur:
        lst_value.append(row[0])
        lst_count.append(row[1])
        lst_nc.append(row[2])
        lst_nid_2.append(row[3])                   
        del row

for i in range(len(lst_nid_2)):
    if lst_value[i]!=i+1:
        lst_value.insert(i,0)
    


h=0
for k in range (len(lst_nid_2)):
   
    if lst_value[k]!=lst_nid_2[k]:
        
        d=lst_count.insert(lst_nid_2[k]-1,0)
        
with arcpy.da.UpdateCursor(total_exp,"total") as ucur:
            for row in ucur:
                row[0]=lst_count[h]
                ucur.updateRow(row)
                h=h+1
                del row
##...........................................................................
arcpy.AddField_management(train_exp,"train","DOUBLE")
arcpy.AddField_management(train_exp,"NID","LONG")

block="""rec=0
def yaz():
    global rec
    pstart=1
    pinterval=1
    if(rec==0):
        rec=pstart
    else:
        rec+=pinterval
    return rec"""
expression="yaz()"
arcpy.CalculateField_management(train_exp,"NID",expression,"PYTHON",block)
lst_nid=list()

with arcpy.da.SearchCursor(train_exp,"NID") as dd:
    for row in dd:
        lst_nid.append(row[0])
        
        del row
    
    del dd

mx=max(lst_nid)
crs=arcpy.da.InsertCursor(train_exp,["NID"])
for i in range(mx+1,101):
   crs.insertRow("0")
arcpy.CalculateField_management(train_exp,"NID",expression,"PYTHON",block)
lst_value=[]
lst_count=[]
lst_nc=[]
lst_nid_2=[]
sc_fields="value","count","train","NID"
with arcpy.da.SearchCursor(train_exp,sc_fields) as scur:
    for row in scur:
        lst_value.append(row[0])
        lst_count.append(row[1])
        lst_nc.append(row[2])
        lst_nid_2.append(row[3])                   
        del row

for i in range(len(lst_nid_2)):
    if lst_value[i]!=i+1:
        lst_value.insert(i,0)
    


h=0
for k in range (len(lst_nid_2)):
   
    if lst_value[k]!=lst_nid_2[k]:
        
        d=lst_count.insert(lst_nid_2[k]-1,0)
        

with arcpy.da.UpdateCursor(train_exp,"train") as ucur:
            for row in ucur:
                row[0]=lst_count[h]
                ucur.updateRow(row)
                h=h+1
                del row
##...........................................................
arcpy.AddField_management(test_exp,"test","DOUBLE")
arcpy.AddField_management(test_exp,"NID","LONG")

block="""rec=0
def yaz():
    global rec
    pstart=1
    pinterval=1
    if(rec==0):
        rec=pstart
    else:
        rec+=pinterval
    return rec"""
expression="yaz()"
arcpy.CalculateField_management(test_exp,"NID",expression,"PYTHON",block)
lst_nid=list()

with arcpy.da.SearchCursor(test_exp,"NID") as dd:
    for row in dd:
        lst_nid.append(row[0])
        
        del row
    
    del dd

mx=max(lst_nid)
crs=arcpy.da.InsertCursor(test_exp,["NID"])
for i in range(mx+1,101):
   crs.insertRow("0")
arcpy.CalculateField_management(test_exp,"NID",expression,"PYTHON",block)
lst_value=[]
lst_count=[]
lst_nc=[]
lst_nid_2=[]
sc_fields="value","count","test","NID"
with arcpy.da.SearchCursor(test_exp,sc_fields) as scur:
    for row in scur:
        lst_value.append(row[0])
        lst_count.append(row[1])
        lst_nc.append(row[2])
        lst_nid_2.append(row[3])                   
        del row

for i in range(len(lst_nid_2)):
    if lst_value[i]!=i+1:
        lst_value.insert(i,0)
    


h=0
for k in range (len(lst_nid_2)):
   
    if lst_value[k]!=lst_nid_2[k]:
        
        d=lst_count.insert(lst_nid_2[k]-1,0)
        


with arcpy.da.UpdateCursor(test_exp,"test") as ucur:
            for row in ucur:
                row[0]=lst_count[h]
                ucur.updateRow(row)
                h=h+1
                del row
##..........................................................................


arcpy.JoinField_management(total_exp,"NID",train_exp,"NID","train")
arcpy.JoinField_management(total_exp,"NID",test_exp,"NID","test")
##///////////////////////Sum of Cumulative////////////////////////////////////

arcpy.AddField_management(total_exp,"kum_total","DOUBLE")
arcpy.AddField_management(total_exp,"kum_train","DOUBLE")
arcpy.AddField_management(total_exp,"kum_test","DOUBLE")

block2="""rec=0
def kum_tot(r):
    global rec
    pstart=r
    pinterval=r
    if(rec==0):
        rec=pstart
    else:
        rec+=pinterval
    return rec"""
expression2="kum_tot(!total!)"
arcpy.CalculateField_management(total_exp,"kum_total",expression2,"PYTHON",block2)
arcpy.CalculateField_management(total_exp,"kum_train","kum_tot(!train!)","PYTHON",block2)
arcpy.CalculateField_management(total_exp,"kum_test","kum_tot(!test!)","PYTHON",block2)
tot_fields="kum_total","kum_train","kum_test"
lst_tot=[]
lst_tr=[]
lst_tst=[]
with arcpy.da.SearchCursor(total_exp,tot_fields) as scur2:
    for row in scur2:
        lst_tot.append(row[0])
        lst_tr.append(row[1])
        lst_tst.append(row[2])
        del row
    del scur2

toplam_tot=max(lst_tot)
toplam_tr=max(lst_tr)
toplam_tst=max(lst_tst)

##......................................................................
arcpy.AddField_management(total_exp,"c_tot","DOUBLE")
arcpy.AddField_management(total_exp,"c_tr","DOUBLE")
arcpy.AddField_management(total_exp,"c_tst","DOUBLE")
c="kum_total","kum_train","kum_test","c_tot","c_tr","c_tst"
with arcpy.da.UpdateCursor(total_exp,c) as ucur2:
    for row in ucur2:
        v=row[0]/toplam_tot
        k=row[1]/toplam_tr
        l=row[2]/toplam_tst
        row[3]=1-v
        row[4]=1-k
        row[5]=1-l
        ucur2.updateRow(row)
y="c_tot","c_tr","c_tst"
tot=[]
tr=[]
ts=[]
with arcpy.da.SearchCursor(total_exp,y) as scur2:
    for row in scur2:
        tot.append(row[0])
        tr.append(row[1])
        ts.append(row[2])
        del row
    del scur2

tot.insert(0,1)
tr.insert(0,1)
ts.insert(0,1)

tr_son=[]
ts_son=[]
for i in range(100):
    b=tot[i]-tot[i+1]
    n=tr[i]
    m=ts[i]
    p=b*n
    t=b*m
    tr_son.append(p)
    ts_son.append(t)
f=round(sum(tr_son)*100,2)
g=round(sum(ts_son)*100,2)

arcpy.AddMessage("Success rate is: {}".format(sum(tr_son)*100))
arcpy.AddMessage("prediction rate is: {}".format(sum(ts_son)*100))
#///////////////////////////////AUC graph is plotting//////////////////////////
sc=plt.plot(tot,tr,color="red",label=":Success Rate"+"("+str(f)+")")

pr=plt.plot(tot,ts,color="blue",label=":Prediction Rate"+"("+str(g)+")")

plt.xlabel("1-Specifity")
plt.ylabel("Sensitivity")
plt.legend(loc="lower right")
arcpy.AddMessage("AUC Graph is saved as auc_mlp.png")
auc_graph=os.path.join(sf,"auc_mlp.png")
plt.savefig(auc_graph,dpi=150)
plt.close("all")
#///////////////////////////////Results is saving as .txt file/////////////////
from datetime import datetime
zaman=datetime.now()
os.chdir(sf)
if os.path.isfile("MLP.txt"):
    file=open("MLP.txt","a+")
    file.write("MULTI LAYER PERCEPTERON ALGORITHM\n")
    file.write("Date-:---------------------------:{}\n".format(zaman))
    file.write("Save folder:---------------------:{}\n".format(sf))
    file.write("Hidden layer size:---------------:{}\n".format(hls))
    file.write("Activation function:-------------:{}\n".format(act))
    file.write("Solver:--------------------------:{}\n".format(slv))
    file.write("Alpha----------------------------:{}\n".format(alpha))
    file.write("Learning rate:-------------------:{}\n".format(l_rate))
    file.write("Learning rate init---------------:{}\n".format(l_rate_init))
    file.write("Max iter:------------------------:{}\n".format(max_it))
    file.write("Momentum-------------------------:{}\n".format(mom))
    file.write("Success rate:--------------------:{}\n".format(f))
    file.write("Predict rate:--------------------:{}\n".format(g))
    file.close()
else:
    file=open("MLP.txt","w+")
    file.write("MULTI LAYER PERCEPTERON ALGORITHM\n")
    file.write("Date-:---------------------------:{}\n".format(zaman))
    file.write("Save folder:---------------------:{}\n".format(sf))
    file.write("Hidden layer size:---------------:{}\n".format(hls))
    file.write("Activation function:-------------:{}\n".format(act))
    file.write("Solver:--------------------------:{}\n".format(slv))
    file.write("Alpha----------------------------:{}\n".format(alpha))
    file.write("Learning rate:-------------------:{}\n".format(l_rate))
    file.write("Learning rate init---------------:{}\n".format(l_rate_init))
    file.write("Max iter:------------------------:{}\n".format(max_it))
    file.write("Momentum-------------------------:{}\n".format(mom))
    file.write("Success rate:--------------------:{}\n".format(f))
    file.write("Predict rate:--------------------:{}\n".format(g))
    file.close()

arcpy.ClearWorkspaceCache_management()
arcpy.AddMessage("FINISHED")
