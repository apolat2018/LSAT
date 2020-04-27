# -*- coding: cp1254 -*-
#Landslide susceptibility analysis data preparation Script
#This script prepares data with the methods of frequency ratio and information value to create Landslide Susceptibility Map

#Ali POLAT (2018)

#////////////////////IMPORTING THE REQUIRED LIBRARIES/////////////////////////
import arcpy,math
import os
from arcpy.sa import *
arcpy.env.overwriteOutput = True
import numpy as np
import pandas as pd

#////////////////////////////Getting Input Parameters//////////////////////////
rec=arcpy.GetParameterAsText(0)##reclassified data(parameters)
hey_ham=arcpy.GetParameterAsText(1)##landslides(shp)
area=arcpy.GetParameterAsText(2)##area(shp)#"
cell_size=arcpy.GetParameterAsText(3)#Cell size of the raster map
split_size=arcpy.GetParameterAsText(4)#Train-Test split size
arcpy.env.workspace=rec
rec_list1=arcpy.ListDatasets("rec*","raster")#The first three letter of Recclassed data name must be  as rec
arcpy.AddMessage(rec_list1)
os.chdir(rec)
ext_file=rec_list1[0]
ext=os.path.join(rec,ext_file)
arcpy.env.extent=ext
##////////////////////////////////////Starting analysis////////////////////////
arcpy.AddMessage("Starting DATA PREPARATION...")
arcpy.AddMessage("{} raster data imported as parameter".format(len(rec_list1)))

for i in rec_list1:
    ad=i.split("_")
    clipped="clp_"+ad[1]
    arcpy.Clip_management(i,"",clipped,area,"","ClippingGeometry","MAINTAIN_EXTENT")

rec_list=arcpy.ListDatasets("clp_*","raster")#Clipped raster file
##//////////////////////////PREDICTION DATA IS CREATING////////////////////////
fields= [c.name for c in arcpy.ListFields(area)]
field=fields[0]
arcpy.AddMessage("Area is converting to Raster")   

hepsi=arcpy.PolygonToRaster_conversion(area,field,"hepsi","CELL_CENTER","",cell_size)
arcpy.AddMessage("Raster is converting to Point")

all_poi=arcpy.RasterToPoint_conversion(hepsi,"all_poi.shp","VALUE")
arcpy.AddField_management(all_poi,"class","LONG")
fields0= [c.name for c in arcpy.ListFields(all_poi) if not c.required]
fields0.remove("class")
##print fields0
arcpy.DeleteField_management(all_poi,fields0)

##//////////////////PREPARING TRAIN AND VALIDATION DATA////////////////////////
arcpy.AddMessage("PREPARING TRAIN AND VALIDATION DATA")
arcpy.AddMessage("Clipping landslides")
arcpy.Clip_analysis(hey_ham,area,"hey.shp")
hey="hey.shp"
fields1= [c.name for c in arcpy.ListFields(hey)]
field1=fields1[0]
arcpy.AddMessage("landslide pixels is converting to point")
arcpy.SubsetFeatures_ga(hey,"train_1.shp","test_1.shp",split_size,"PERCENTAGE_OF_INPUT")###Splitting landslides as %70 train and %30 test
hey_ras=arcpy.PolygonToRaster_conversion("train_1.shp",field1,"hey_ras","CELL_CENTER","",cell_size)
hey_ras_test=arcpy.PolygonToRaster_conversion("test_1.shp",field1,"hey_ras_test","CELL_CENTER","",cell_size)
hey_poi=arcpy.RasterToPoint_conversion("hey_ras","hey_poi.shp","VALUE")
hey_poi_test=arcpy.RasterToPoint_conversion("hey_ras_test","hey_poi_test.shp","VALUE")

##///////////////////////////////TRAIN DATA////////////////////////////////////
arcpy.AddField_management("hey_poi.shp","class","SHORT")
fields2= [c.name for c in arcpy.ListFields("hey_poi.shp") if not c.required]
fields2.remove("class")
arcpy.DeleteField_management("hey_poi.shp",fields2)##deleted all field except class

with arcpy.da.UpdateCursor("hey_poi.shp","class") as upcur2:###assigned 1 vale to class 
    for row in upcur2:
        row[0]=1
        upcur2.updateRow(row)

###//////////////////////////////VALIDATION DATA///////////////////////////////     
arcpy.AddField_management("hey_poi_test.shp","class","SHORT")
fields5= [c.name for c in arcpy.ListFields("hey_poi_test.shp") if not c.required]
fields5.remove("class")
arcpy.DeleteField_management("hey_poi_test.shp",fields5)##deleted all field except class

with arcpy.da.UpdateCursor("hey_poi_test.shp","class") as upcur5:###assigned 1 vale to class 
    for row in upcur5:
        row[0]=1
        upcur5.updateRow(row)

#/////////////////calculating how many pixel in train and test data////////////
train_count = int(arcpy.GetCount_management("hey_poi.shp").getOutput(0))
test_count = int(arcpy.GetCount_management("hey_poi_test.shp").getOutput(0))

##//////////////////////selecting no landsllide pixels/////////////////////////
no_ls_trn=train_count
no_ls_tst=test_count
arcpy.Erase_analysis("all_poi.shp",hey,"erased_poi.shp")##erased landslide pixels in the All pix
arcpy.SubsetFeatures_ga("erased_poi.shp","data_trn.shp","",no_ls_trn,"ABSOLUTE_VALUE")###random no landslide pixels is selecting as many as train landslide pixels
arcpy.SubsetFeatures_ga("erased_poi.shp","data_tst.shp","",no_ls_tst,"ABSOLUTE_VALUE")###random no landslide pixels is selecting as many as validation landslide pixels

merged_train="hey_poi.shp","data_trn.shp"
merged_test="hey_poi_test.shp","data_tst.shp"

arcpy.Merge_management(merged_train,"train.shp")#Train data
arcpy.Merge_management(merged_test,"validation.shp")#Validation data
#//////////////////CALCULATING FREQUENCY RATIO AND INFORMATION VALUE///////////
analiz_hey="train_1.shp"
arcpy.AddMessage("Extracting landslide pixels")
for i in rec_list:
    masking=ExtractByMask(i,analiz_hey)
    mask_out = os.path.join(rec,str("ext"+(i)))
    masking.save(mask_out)
    arcpy.AddMessage(i+" is processing")
hey_list=arcpy.ListDatasets("ext*","Raster")
for n in hey_list:
    d=[]
    fields0= [c.name for c in arcpy.ListFields(n) if not c.required]

    for k in fields0:
        if k=="VALUE" or k=="COUNT" :
            d.append(k)

        else:

            pass
        
    if len(fields0)>2 and len(d)==2:
        fields0.remove(d[0])
        fields0.remove(d[1])
        arcpy.DeleteField_management(n,fields0)
    d=[]

for n in rec_list:
    d=[]
    fields0= [c.name for c in arcpy.ListFields(n) if not c.required]

    for k in fields0:
        if k=="VALUE" or k=="COUNT" :
            d.append(k)

        else:

            pass
        
    if len(fields0)>2 and len(d)==2:
        fields0.remove(d[0])
        fields0.remove(d[1])
        arcpy.DeleteField_management(n,fields0)
    
#Creating raster  files with Normalized frequency ratio and Normalized information value

for j,k in zip(rec_list,hey_list):
    arcpy.JoinField_management(j,"Value",k,"Value","count")
lst=[]
lst2=[]
lstmax=[]
lstmin=[]
top_pix_field="count","hp"
fieldTpx=['sumtpx','sumlpx']
max_min="max","min"
max_min_iv="max_iv","min_iv"
for l in rec_list:
    arcpy.AddMessage("Creating raster  files with Normalized frequency ratio and Normalized information value")
    ad=l.split("_")
    clipped="clp_"+ad[1]
    arcpy.AddMessage("{} is finished".format(clipped))
    outname =str("t_"+(ad[1])+".dbf")
    outname_table=os.path.join(rec,outname)
    outname_to_sus_fr=os.path.join(rec,str("fr_"+(ad[1])))
    outname_to_sus_iv=os.path.join(rec,str("iv_"+(ad[1])))
    arcpy.AddField_management(l,"hp","DOUBLE")
    arcpy.AddField_management(l,"sumtpx","DOUBLE")
    arcpy.AddField_management(l,"sumlpx","DOUBLE")
    arcpy.AddField_management(l,"max","DOUBLE")
    arcpy.AddField_management(l,"min","DOUBLE")
    arcpy.AddField_management(l,"max_iv","DOUBLE")
    arcpy.AddField_management(l,"min_iv","DOUBLE")
    arcpy.AddField_management(l,"fr","DOUBLE")
    arcpy.AddField_management(l,"nfr","SHORT")
    arcpy.AddField_management(l,"iv","DOUBLE")
    arcpy.AddField_management(l,"niv","SHORT")
    arcpy.CalculateField_management(l,"hp","!count_1!","PYTHON")
    
    ivv="hp","iv","COUNT","SUMLPX","SUMTPX","niv","nfr"
    with arcpy.da.UpdateCursor(l,"hp") as upcursor:
        for row in upcursor:
            if row[0]== None:
                row[0]=0
                upcursor.updateRow(row)
            del row
        del upcursor    
    with arcpy.da.SearchCursor(l,top_pix_field) as cursor:
        sum_tpx=0
        sum_lpx=0
        for row1 in cursor:
            sum_tpx +=row1[0]
            sum_lpx +=row1[1]
        lst.append(sum_tpx)
        lst2.append(sum_lpx)
        del row1
    del cursor  
    with arcpy.da.UpdateCursor(l,fieldTpx) as upcursor2:
        for row2 in upcursor2:
            row2[0]=lst[-1]
            row2[1]=lst2[-1]
            upcursor2.updateRow(row2)
            del row2
        del upcursor2  
    arcpy.CalculateField_management(l,"fr","(!hp!/!sumlpx!)/(!COUNT!/!sumtpx!)","PYTHON")
    maximum = max(row3[0] for row3 in arcpy.da.SearchCursor(l, ['fr']))
    minumum = min(row3[0] for row3 in arcpy.da.SearchCursor(l, ['fr']))
    with arcpy.da.SearchCursor(l,"fr") as cursor2:
        for row3 in cursor2:
            lstmax.append(row3[0])
            lstmin.append(row3[0])
            maxfr=max(lstmax)
            minfr=min(lstmin)
            del row3
        del cursor2  
    with arcpy.da.UpdateCursor(l,max_min) as upcursor3:
        for row4 in upcursor3:
            row4[0]=maximum
            row4[1]=minumum
            upcursor3.updateRow(row4)
            del row4
        del upcursor3  
    arcpy.CalculateField_management(l,"nfr","((!fr!-!min!)/(!max!-!min!))*100","PYTHON")
    with arcpy.da.UpdateCursor(l,ivv) as upcursor3:
        for row4 in upcursor3:
            if row4[0]==0:
                row4[1]=0
            else:
                row4[1]=math.log((row4[0]/row4[2])/(row4[3]/row4[4]))
            upcursor3.updateRow(row4)
            del row4
        del upcursor3  
    
    maximum_iv = max(row5[0] for row5 in arcpy.da.SearchCursor(l, ["iv"]))
    minumum_iv = min(row5[0] for row5 in arcpy.da.SearchCursor(l, ["iv"]))
    with arcpy.da.UpdateCursor(l,max_min_iv) as upcursor4:
        for row6 in upcursor4:
            row6[0]=maximum_iv
            row6[1]=minumum_iv
            upcursor4.updateRow(row6)
            del row6
        del upcursor4  
    arcpy.CalculateField_management(l,"niv","((!iv!-!min_iv!)/(!max_iv!-!min_iv!))*100","PYTHON")
    with arcpy.da.UpdateCursor(l,ivv) as upcursor3:
        for row4 in upcursor3:
            if row4[0]==0:
                row4[5]=0
                row4[6]=0
            upcursor3.updateRow(row4)
            del row4
        del upcursor3 
    to_sus_fr=Lookup(l,"nfr")
    to_sus_iv=Lookup(l,"niv")
    to_sus_fr.save(outname_to_sus_fr)
    to_sus_iv.save(outname_to_sus_iv)

#/////////////////////////Exporting analysis data as csv///////////////////////
n_list_fr=arcpy.ListDatasets("fr_*","Raster")
n_list_iv=arcpy.ListDatasets("iv_*","Raster")
arcpy.Copy_management("train.shp","train_iv.shp")
arcpy.Copy_management("validation.shp","validation_iv.shp")
arcpy.Copy_management("all_poi.shp","all_poi_iv.shp")
arcpy.AddMessage("Preparing Data for Frequency ratio value")
    
ExtractMultiValuesToPoints("train.shp", n_list_fr,"NONE")
ExtractMultiValuesToPoints("validation.shp", n_list_fr,"NONE")
arcpy.AddMessage("Saving TRAINING data as train_fr.csv")
arcpy.AddMessage("Saving validation data as valid_fr.csv")
arcpy.TableToTable_conversion("train.shp",rec,"train_fr.csv")
arcpy.TableToTable_conversion("validation.shp",rec,"valid_fr.csv")

arcpy.AddMessage("Extracting Values to Point")
ExtractMultiValuesToPoints("all_poi.shp", n_list_fr,"NONE")
arcpy.AddMessage("Extracting finished")

arcpy.AddXY_management("all_poi.shp")
arcpy.AddMessage("Saving ALL data as pre_fr.csv")
arcpy.TableToTable_conversion("all_poi.shp",rec,"pre_fr.csv")

arcpy.AddMessage("Preparing Data for information  value")
###.............................INFORMATION VALUE..................
ExtractMultiValuesToPoints("train_iv.shp", n_list_iv,"NONE")
ExtractMultiValuesToPoints("validation_iv.shp", n_list_iv,"NONE")
arcpy.AddMessage("Saving TRAINING data as train_iv.csv")
arcpy.AddMessage("Saving validation data as valid_iv.csv")
arcpy.TableToTable_conversion("train_iv.shp",rec,"train_iv.csv")
arcpy.TableToTable_conversion("validation_iv.shp",rec,"valid_iv.csv")

arcpy.AddMessage("Extracting Values to Point")
ExtractMultiValuesToPoints("all_poi_iv.shp", n_list_iv,"NONE")
arcpy.AddMessage("Extracting finished")

arcpy.AddXY_management("all_poi_iv.shp")
arcpy.AddMessage("Saving ALL data as pre_iv.csv")
arcpy.TableToTable_conversion("all_poi_iv.shp",rec,"pre_iv.csv")
arcpy.AddMessage("Validation landslide was saved as test_1.shp as polygon type")
arcpy.ClearWorkspaceCache_management()  
## ///////////////////////////////////FINISHED///////////////////////////////// 
    


