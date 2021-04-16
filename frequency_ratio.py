# -*- coding: cp1254 -*-
#This script requires data created with Data preparation tool.
#You have to create data with normalized frequency ratio before using this tool.
#So, firstly you should create data with data preparation tool .
#//////////////////////////////Ali POLAT (2018)////////////////////////////////

#////////////////////IMPORTING THE REQUIRED LIBRARIES/////////////////////////
arcpy.AddMessage("Analysis started")
import arcpy
import os,sys
from arcpy.sa import *
from matplotlib import pyplot as plt
arcpy.env.overwriteOutput = True
#////////////////////////////Getting Input Parameters//////////////////////////
rec=arcpy.GetParameterAsText(0)#The folder where the Parameter files (Reclassed raster data) are located.
sf=arcpy.GetParameterAsText(1)#saving folder
koordinat=arcpy.GetParameterAsText(2)#Coordinate system
cell_size=arcpy.GetParameterAsText(3)#Cell size

arcpy.env.workspace=rec
#////////////////////////////////////Starting Analysis/////////////////////////
arcpy.AddMessage("Starting Frequency Ratio Analysis...")

os.chdir(rec)
duy_lst=arcpy.ListDatasets("fr*","Raster")
sus_map=arcpy.sa.CellStatistics(duy_lst,"SUM","DATA")
sus_map_name=os.path.join(sf,str("fr_sus"))
sus_map.save(sus_map_name)
#///Analysis finished anad LSM is saving to sf folder as iv_sus raster file////
arcpy.AddMessage("Analysis finished")
arcpy.AddMessage("Susceptibility Map was created in {} folder as fr_sus raster file".format(sf))

#//////////////////////////CALCULATING PERFORMANCE/////////////////////////////
arcpy.AddMessage("ROC calculation is starting.....")
mx=float (arcpy.GetRasterProperties_management (sus_map, "MAXIMUM").getOutput (0))
mn=float (arcpy.GetRasterProperties_management (sus_map, "MINIMUM").getOutput (0))

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

total=Reclassify(sus_map,"VALUE",RemapRange(d),"NODATA")
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
##/////////////////////////Calculataing sum of Cumulative//////////////////////
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
arcpy.AddMessage(len(tot))
arcpy.AddMessage(tr)

plt.plot(tot,tr,color="red",label=":Success Rate"+"("+str(f)+")")#success rate

plt.plot(tot,ts,color="blue",label=":Prediction Rate"+"("+str(g)+")")#prediction rate

plt.xlabel("1-Specifity")
plt.ylabel("Sensitivity")
plt.legend(loc="lower right")

arcpy.AddMessage("AUC Graph is saved as auc_fr.png")
auc_graph=os.path.join(sf,"auc_fr.png")
plt.savefig(auc_graph,dpi=150)

plt.close("all")

arcpy.ClearWorkspaceCache_management() 
arcpy.AddMessage("FINISHED")
#////////////////////FINISHED//////////////////////////////////////////////////
