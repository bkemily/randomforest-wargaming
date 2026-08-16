import numpy as np
from pyspark.sql.functions import *
from pyspark.ml import Pipeline #not used as of yet, but probably should be
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import traceback
import sys
import datetime
from os.path import getsize

from df_binning import *

headerString =["Start", #localNow
               "dataset", # conn_server_loc
               "key",
               "pct attack", #percent_attack_data,
               "num features",
               "feature_cols",
               "cfsn_mtrx",
               "accuracy",
               "precision",
               "recall",
               "f_measure",
               "areaUnderCurve",
               "truePositive",
               "falsePositive",
               "bin_time",
               "train_time",
               "test_time"]

def printToLog(addMe, logLocation):
    with open(logLocation, 'a') as fd:
        fd.write("" + str(datetime.datetime.now()) + ": " + str(addMe) + "\n")
        
def csvAppendBuffer(*addMe):
    result = '\n'
    for thing in addMe:
        result += '"'
        if (type(thing) == list):
            result += '('
            result += ") (".join(str(itm) for itm in thing)
            result += ')'
        else:
            result += str(thing)
        result += '"'
        result += ','
    #-1 slice to get rid of a trailing ,
    return result[:-1]

def randForestMaster(test, train, binaryClassFlag, bin_time, log_location, rf_results_location, countRuns, localNow, conn_server_loc, key, percent_attack_data, feature_cols):
    rf = RandomForestClassifier(featuresCol = "features", 
                                        labelCol = "label_bin")
              
    begin_randForestTraining = datetime.datetime.now()     
    rfModel = rf.fit(train)
    end_randForestTraining = datetime.datetime.now()
    printToLog("randomForest model fit", log_location)
            
    #############
    # Predictions
    #############   
    #This timing metric currently doesn't make much sense,
    #as it will be significantly impacted by the size of "test"
    begin_randForestPredictions = datetime.datetime.now()
    predictions = rfModel.transform(test)
    end_randForestPredictions = datetime.datetime.now()

    predictions_and_labels = predictions.select(["prediction", 
                                                 "label_bin"])
    predictions_and_labels.selectExpr("cast(prediction as int) prediction")
    metrics = MulticlassMetrics(predictions_and_labels.rdd.map(tuple))
    evaluator = MulticlassClassificationEvaluator(labelCol = "label_bin", 
                                                  predictionCol = "prediction")

    if binaryClassFlag:                   
        binary_metrics = BinaryClassificationMetrics(predictions_and_labels.select("prediction", "label_bin").rdd.map(tuple))
        areaUnderCurve = binary_metrics.areaUnderROC
    else:
        areaUnderCurve = "na"

    cfsn_temp = metrics.confusionMatrix()
    cfsn_mtrx = np.array2string(cfsn_temp.toArray()).replace('\n', '')
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    precision = evaluator.evaluate(predictions, {evaluator.metricName: "precisionByLabel",
                                                 evaluator.metricLabel: 1.0})
    recall = evaluator.evaluate(predictions, {evaluator.metricName: "recallByLabel",
                                              evaluator.metricLabel: 1.0})
    f_measure = evaluator.evaluate(predictions, {evaluator.metricName: "fMeasureByLabel",
                                                 evaluator.metricLabel: 1.0})
    truePositive = evaluator.evaluate(predictions, {evaluator.metricName: "truePositiveRateByLabel",
                                                    evaluator.metricLabel: 1.0})
    falsePositive = evaluator.evaluate(predictions, {evaluator.metricName: "falsePositiveRateByLabel",
                                                     evaluator.metricLabel: 1.0})

    
    train_time = (end_randForestTraining - begin_randForestTraining).total_seconds()
    test_time = (end_randForestPredictions - begin_randForestPredictions).total_seconds()

    printToLog("randomForest metrics finished", log_location)

    ################################
    # Write to save_results_location
    ################################         
    if countRuns:
        buff = csvAppendBuffer(localNow,
                           conn_server_loc,
                           key,
                           percent_attack_data,
                           len(feature_cols),
                           feature_cols,
                           cfsn_mtrx,
                           accuracy,
                           precision,
                           recall,
                           f_measure,
                           areaUnderCurve,
                           truePositive,
                           falsePositive,
                           bin_time,
                           train_time,
                           test_time)
                           
        header = csvAppendBuffer(headerString)
        
        with open(rf_results_location, 'a') as fd:
            if getsize(rf_results_location) == 0:
                fd.write(header)
                
            fd.write(buff)
            fd.close()
            
def gbtMaster(test, train, binaryClassFlag, log_location, bin_time, gb_results_location, countRuns, localNow, conn_server_loc, key, percent_attack_data, feature_cols):
    gbt = GBTClassifier(featuresCol = "features",
                        labelCol = "label_bin")
                
    begin_gbTraining = datetime.datetime.now()
    gbtModel = gbt.fit(train)
    end_gbTraining = datetime.datetime.now()

    printToLog("GBTClassifier model fit", log_location)
    
    #############
    # Predictions
    #############   
    #This timing metric currently doesn't make much sense,
    #as it will be significantly impacted by the size of "test"
    begin_gbPredictions = datetime.datetime.now()
    gbPredictions = gbtModel.transform(test)
    end_gbPredictions = datetime.datetime.now()

    gbPredictions_and_labels = gbPredictions.select(["prediction", 
                                                     "label_bin"])
    gbPredictions_and_labels.selectExpr("cast(prediction as int) prediction")
    gbMetrics = MulticlassMetrics(gbPredictions_and_labels.rdd.map(tuple))
    gbEval = MulticlassClassificationEvaluator(labelCol = "label_bin", 
                                               predictionCol = "prediction")                                 
    gb_binary_metrics = BinaryClassificationMetrics(gbPredictions_and_labels.select("prediction", "label_bin").rdd.map(tuple))
    
    gb_areaUnderCurve = gb_binary_metrics.areaUnderROC
    gb_cfsn_temp = gbMetrics.confusionMatrix()
    gb_cfsn_mtrx = np.array2string(gb_cfsn_temp.toArray()).replace('\n', '')
    gb_accuracy = gbEval.evaluate(gbPredictions, {gbEval.metricName: "accuracy"})
    gb_precision = gbEval.evaluate(gbPredictions, {gbEval.metricName: "precisionByLabel",
                                                 gbEval.metricLabel: 1.0})
    gb_recall = gbEval.evaluate(gbPredictions, {gbEval.metricName: "recallByLabel",
                                              gbEval.metricLabel: 1.0})
    gb_f_measure = gbEval.evaluate(gbPredictions, {gbEval.metricName: "fMeasureByLabel",
                                                 gbEval.metricLabel: 1.0})
    gb_truePositive = gbEval.evaluate(gbPredictions, {gbEval.metricName: "truePositiveRateByLabel",
                                                    gbEval.metricLabel: 1.0})
    gb_falsePositive = gbEval.evaluate(gbPredictions, {gbEval.metricName: "falsePositiveRateByLabel",
                                                     gbEval.metricLabel: 1.0})

    gb_train_time = (begin_gbTraining - end_gbTraining).total_seconds()
    gb_test_time = (begin_gbPredictions - end_gbPredictions).total_seconds()
    ################################
    # Write to save_results_location
    ################################
    if countRuns:
        gbuff = csvAppendBuffer(localNow,
                           conn_server_loc,
                           key,
                           percent_attack_data,
                           len(feature_cols),
                           feature_cols,
                           gb_cfsn_mtrx,
                           gb_accuracy,
                           gb_precision,
                           gb_recall,
                           gb_f_measure,
                           gb_areaUnderCurve,
                           gb_truePositive,
                           gb_falsePositive,
                           bin_time,
                           gb_train_time,
                           gb_test_time)
                           
        header = csvAppendBuffer(headerString)
        
        with open(gb_results_location, 'a') as fd:
            if getsize(gb_results_location) == 0:
                fd.write(header)
                
            fd.write(gbuff)
            fd.close()
    
    printToLog("GBTClassifier metrics finished", log_location)