# -*- coding: utf-8 -*-
"""
Created on Wed May 18 02:56:50 2022

@author: Tom
"""

import numpy as np
import builtins
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.functions as func
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import StringIndexer,VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import traceback
import sys
import datetime
from os.path import getsize

from df_binning import *
from generate_individual_labels_dataframe import *
from TM_ML import *

#############
# Run variables
#############

countRuns = True
binaryClassFlag = False
runRF = True
runGBT = False
numAttributes = 18

benign_label = "none"
label_col_name = "label_multi"
percent_attack_data = 0.3
num_top_labels = 10
not_enough_rows_threshhold = 500

conn_server_loc = '/home/kali/datasets/zeekdata24_full_csv'
rf_results_location = '/home/kali/datasets/randomforest/results/randomForest/' + str(datetime.date.today()) + '_results.csv'
gb_results_location = '/home/kali/datasets/randomforest/results/gradientBoost/' + str(datetime.date.today()) + '_results.csv'
log_location = "/home/kali/datasets/randomforest/logs/randomForest/" + str(datetime.date.today()) + "_log.txt"
    
########################
# code start
########################

spark = (
    SparkSession
    .builder
    .master("local[*]")
    .appName("TM_RandoForest")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
    )
        
sc = spark.sparkContext    

localNow = str(datetime.datetime.now().strftime("%H:%M:%S"))
printToLog("--------------\n-------------\nBegin run\n-----------\n----------\n\n", log_location)
printToLog("Dataset location - " + conn_server_loc, log_location)

source_df = spark.read.option("header", True).options(inferSchema=True).csv(conn_server_loc)
 
try:
    df_renamed = rename_columns(source_df)
except Exception as e:
    printToLog("Error:  " + str(e), log_location)
    print(traceback.format_exc())    

if (binaryClassFlag):
    df_dict = create_dict_of_dfs(df_renamed, benign_label, label_col_name, percent_attack_data, num_top_labels, True, log_location)
else:
    df_dict = { "All label techniques" : df_renamed }
    printToLog("df prepared for multiclass problem", log_location)

printToLog("", log_location) 

for key in df_dict:
    printToLog("Starting df " + key, log_location)
    printToLog("Instance start-time to match csv record: " + localNow, log_location)
    #Create df of interest (a certain label technique)
    unbinned_df = df_dict[key]
    
    #reject dataframes that are too small
    if unbinned_df.count() < not_enough_rows_threshhold:
        printToLog("df " + key + " contains only " + str(unbinned_df.count()) +
                   " items, analysis discarded by row threshold\n\n", log_location)
    #bin dataframes that are not rejected
    else:
        printToLog(str(unbinned_df.count()) + " rows", log_location)
        printToLog("Binning start", log_location)

        begin_binning = datetime.datetime.now()
        
        if binaryClassFlag:
            unbinned_df = unbinned_df.withColumn("label_bin", when((col("label_multi") != "none"), 0.0).otherwise(1.0))
        else:
            # exclude benign records entirely — no 8th "none" bin for benign traffic.
            unbinned_df = unbinned_df.filter(col("label_multi") != benign_label)

            # downsample every tactic to match the smallest tactic's count,
            # so no single t-code (e.g. Credential Access) dominates training.
            class_counts = unbinned_df.groupBy("label_multi").count().collect()
            min_count = builtins.min(row["count"] for row in class_counts)
            printToLog(f"Balancing all tactics down to {min_count} rows each", log_location)

            balanced_dfs = []
            for row in class_counts:
                tactic, tactic_count = row["label_multi"], row["count"]
                fraction = min_count / tactic_count
                tactic_df = unbinned_df.filter(col("label_multi") == tactic)
                if fraction < 1.0:
                    tactic_df = tactic_df.sample(withReplacement=False, fraction=fraction, seed=1234)
                balanced_dfs.append(tactic_df)

            unbinned_df = balanced_dfs[0]
            for df in balanced_dfs[1:]:
                unbinned_df = unbinned_df.unionByName(df)

            unbinned_df = genNominalBinnedDF(unbinned_df, "label_multi", 1.0, False, mapping_output_path="/home/kali/datasets/randomforest/logs/randomForest/" + str(datetime.date.today()) + "_label_mapping")
            unbinned_df = unbinned_df.withColumn("label_bin", unbinned_df["label_multi_bin"].cast('double') )
            
        #Carry out the binning on the df
        numeric_percent_trim = 0.02
        nominal_percent_agg = 0.98
        replace_bool = False

        
        attrList_ip_addr = ['dest_ip','src_ip']
        attrList_port = ['dest_port', 'src_port']
        attrList_bool = ['local_orig', 'local_resp']
        attrList_nominal = ['protocol', 'conn_state', 'history','service']
        attrList_numeric = ['duration','orig_bytes','orig_pkts','orig_ip_bytes','resp_bytes','resp_pkts','resp_ip_bytes','missed_bytes']
        
        # %%
        conn_df = genFullBinnedDF(unbinned_df, attrList_ip_addr, attrList_port, attrList_bool, attrList_nominal, nominal_percent_agg, attrList_numeric, numeric_percent_trim, replace_bool).persist()
        printToLog("Binning finished", log_location)

        # balanced class weights rarer classes get a higher weight so the
        # model is penalized more for missing them. Same formula sklearn uses for
        # class_weight='balanced': total / (num_classes * class_count).
        class_counts = conn_df.groupBy("label_bin").count()
        total_count = conn_df.count()
        num_classes = class_counts.count()
        class_counts = class_counts.withColumn("class_weight", total_count / (num_classes * col("count")))
        conn_df = conn_df.join(class_counts.select("label_bin", "class_weight"), on="label_bin", how="left").persist()
        
        end_binning = datetime.datetime.now()
        bin_time = (end_binning - begin_binning).total_seconds()
        
        attribute_size_list = [18, 12, 9, 6]
        attribute_list = (
                        "history_bin",
                        "protocol_bin",
                        "service_bin",
                        "orig_bytes_bin",
                        "dest_ip_bin",
                        "orig_pkts_bin",
                        "orig_ip_bytes_bin",
                        "local_resp_bin",
                        "dest_port_bin",
                        "duration_bin",
                        "conn_state_bin",
                        "resp_pkts_bin",
                        "resp_ip_bytes_bin",
                        "src_port_bin",
                        "resp_bytes_bin",
                        "src_ip_bin",
                        "local_orig_bin",
                        "missed_bytes_bin",
                          )
        confusion_matrices = []

        for attNum in attribute_size_list:
            feature_cols = []
            local_att_list = attribute_list[:attNum]
            for clm, types in conn_df.dtypes:
                if str(clm) in local_att_list:
                    feature_cols.append(clm)
    
            assembler = VectorAssembler(inputCols = feature_cols, outputCol = "features")
            features_df = assembler.transform(conn_df)
    
            printToLog("feature_cols - " + str(len(feature_cols)) + " - " + str(feature_cols), log_location)
    
            train, test = features_df.randomSplit([0.7, 0.3], seed = 2057)
                                                      
            for clm in feature_cols:
                printToLog(str(clm) + " distinct values: " + str(train.select(clm).distinct().count()), log_location)
    
            train.printSchema()
    
            if runRF:
                tactic_results = randForestMaster(test, train, binaryClassFlag, bin_time, log_location, rf_results_location, countRuns, localNow, conn_server_loc, key, percent_attack_data, feature_cols, weight_col="class_weight")
                confusion_matrices.append((attNum, tactic_results))

            if runGBT:
                gbtMaster (test, train, binaryClassFlag, bin_time, log_location,  gb_results_location, countRuns, localNow, conn_server_loc, key, percent_attack_data, feature_cols)

            print("\n\nEnd of for each loop " + key + "\n\n\n\n\n\n\n")
            printToLog("End of for each loop " + key + "\n", log_location)
            conn_df.unpersist()

        # Print results grouped by feature count, then by tactic
        for attNum, tactic_results in confusion_matrices:
            print(f"\n {attNum} features")
            for result in tactic_results:
                print(f"\n{result['tactic_name']}:")
                display(result["two_by_two"])
                print(f"  Accuracy: {result['accuracy']:.1%}   "
                      f"Precision: {result['precision']:.1%}   "
                      f"Recall: {result['recall']:.1%}   "
                      f"F1: {result['f1']:.1%}")

printToLog("End run\n-----------\n----------\n\n", log_location)