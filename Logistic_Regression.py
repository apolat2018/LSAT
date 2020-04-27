# -*- coding: cp1254 -*-
""" 
This script creates Landslide Susceptibility Map (LSM) with Logistic Regression Model 

Ali POLAT (2018)
"""
#////////////////////IMPORTING THE REQUIRED LIBRARIES/////////////////////////
import arcpy
import os
from arcpy.sa import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer,StandardScaler
arcpy.env.overwriteOutput = True
#////////////////////////////Getting Input Parameters//////////////////////////
rec=arcpy.GetParameterAsText(0)#The folder including output data of Data Preparation script
sf=arcpy.GetParameterAsText(1)#output file is saving this Folder
koordinat=arcpy.GetParameterAsText(2)#Coordinate system of map
cell_size=arcpy.GetParameterAsText(3)#Cell size
wt=str(arcpy.GetParameterAsText(4))#weighting data type (Frequency ratio or Information value)
C=arcpy.GetParameterAsText(5)#Regularization Strenght
max_iter=int(arcpy.GetParameterAsText(6))#maximum number of iteration
solver=str(arcpy.GetParameterAsText(7))#Solver
arcpy.env.workspace=rec
rs=float(C)
os.chdir(rec)

#////////////////////////////STARTING ANALYSIS/////////////////////////////////
arcpy.AddMessage("Starting analysis with Logistic Regression algorithm")
os.chdir(rec)
#//////Checking Weighting data Frequency ratio or İnformation value////////////
if wt=="frequency ratio":
    trn="train_fr.csv"
    pre="pre_fr.csv"
    arcpy.AddMessage("Data type= {}".format(wt))
else:
    trn="train_iv.csv"
    pre="pre_iv.csv"
    arcpy.AddMessage("Data type= {}".format(wt))   
#Loading Training data
veriler=pd.read_csv(trn)
veriler=veriler.replace(-9999,"NaN")
#Loading analysis data
analiz=pd.read_csv(pre)
analiz=analiz.replace(-9999,"NaN")
##Preparing parameters
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

pre=np.array(pre)
x=np.array(parametreler)
y=np.array(cls)
#////////////////////////////Results wil be savedas txt file///////////////////
arcpy.AddMessage("Save folder:------------:{}".format(sf))
arcpy.AddMessage("C value:----------------:{}".format(rs))
arcpy.AddMessage("max_iter:---------------:{}".format(max_iter))
arcpy.AddMessage("Solver:-----------------:{}".format(solver))
arcpy.AddMessage("Starting Training")
#///////////////////////////Creating Model/////////////////////////////////////
log_r=LogisticRegression(C=rs,random_state=0,max_iter=max_iter,solver=solver)
#//////////////////////////////////training model////////////////////////////// 
log_r.fit(x,y)
#//////////////////////////////////making prediction///////////////////////////
tahmin=log_r.predict_proba(pre)
logr_cls=pd.DataFrame(data=tahmin,index=range(s_analiz),columns=["zeros","ones"])
K=pd.concat([koor,logr_cls],axis=1)
#///////////////////////////////Saving Prediction Data as log_r.csv////////////
arcpy.AddMessage("Saving Prediction Data as log_r.csv")
logr_result=os.path.join(sf,"logr.csv")
K.to_csv(logr_result,columns=["x","y","ones"])
#//////////////////////////Creating Susceptibility map/////////////////////////
arcpy.AddMessage("Analysis finished")
logr_sus_map=os.path.join(sf,"logr_sus")
arcpy.AddMessage("Creating SUSCEPTIBILITY Map and Calculating ROC ")          
arcpy.MakeXYEventLayer_management(logr_result,"x","y","model",koordinat,"ones")
arcpy.PointToRaster_conversion("model","ones",logr_sus_map,"MOST_FREQUENT","",cell_size)
arcpy.AddMessage("Susceptibility Map was created in {} folder as logr_sus raster file".format(sf))

#////////////////////////////CALCULATING PERFORMANCE///////////////////////////

mx=float (arcpy.GetRasterProperties_management (logr_sus_map, "MAXIMUM").getOutput (0))
mn=float (arcpy.GetRasterProperties_management (logr_sus_map, "MINIMUM").getOutput (0))

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

total=Reclassify(logr_sus_map,"VALUE",RemapRange(d),"NODATA")
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
#///////////////////////////////AUC graph is plotting//////////////////////////
arcpy.AddMessage("Success rate is: {}".format(sum(tr_son)*100))
arcpy.AddMessage("prediction rate is: {}".format(sum(ts_son)*100))
sc=plt.plot(tot,tr,color="red",label=":Success Rate"+"("+str(f)+")")

pr=plt.plot(tot,ts,color="blue",label=":Prediction Rate"+"("+str(g)+")")

plt.xlabel("1-Specifity")
plt.ylabel("Sensitivity")
plt.legend(loc="lower right")
arcpy.AddMessage("AUC Graph is saved as auc_logr.png")
auc_graph=os.path.join(sf,"auc_logr.png")
plt.savefig(auc_graph,dpi=150)
plt.close("all")
#///////////////////////////////Results is saving as .txt file/////////////////
from datetime import datetime
zaman=datetime.now()
os.chdir(sf)
if os.path.isfile("logistic_regression.txt"):
    file=open("logistic_regression.txt","a+")
    file.write("logistic_regression ALGORITHM\n")
    file.write("Date-:------------------:{}\n".format(zaman))
    file.write("Save folder:------------:{}\n".format(sf))
    file.write("C value:----------------:{}\n".format(rs))
    file.write("max_iter:---------------:{}\n".format(max_iter))
    file.write("Solver:-----------------:{}\n".format(solver))
    file.write("Success rate:-----------:{}\n".format(f))
    file.write("Predict rate:-----------:{}\n".format(g))
    file.close()
else:
    file=open("logistic_regression.txt","w+")
    file.write("logistic_regression ALGORITHM\n")
    file.write("Date-:------------------:{}\n".format(zaman))
    file.write("Save folder:------------:{}\n".format(sf))
    file.write("C value:----------------:{}\n".format(rs))
    file.write("max_iter:---------------:{}\n".format(max_iter))
    file.write("Solver:-----------------:{}\n".format(solver))
    file.write("Success rate:-----------:{}\n".format(f))
    file.write("Predict rate:-----------:{}\n".format(g))
    file.close()
    
arcpy.ClearWorkspaceCache_management() 
arcpy.AddMessage("FINISHED")
#///////////////////////////////FINISHED///////////////////////////////////////



