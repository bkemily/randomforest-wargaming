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
               "tp_fp_fn_tn_by_class",
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

def randForestMaster(test, train, binaryClassFlag, bin_time, log_location, rf_results_location, countRuns, localNow, conn_server_loc, key, percent_attack_data, feature_cols, weight_col=None):
    if weight_col:
        rf = RandomForestClassifier(featuresCol="features", labelCol="label_bin", weightCol=weight_col)
    else:
        rf = RandomForestClassifier(featuresCol="features", labelCol="label_bin")
              
    begin_randForestTraining = datetime.datetime.now()     
    rfModel = rf.fit(train)
    end_randForestTraining = datetime.datetime.now()
    printToLog("randomForest model fit", log_location)
            
    begin_randForestPredictions = datetime.datetime.now()
    predictions = rfModel.transform(test)
    end_randForestPredictions = datetime.datetime.now()

    predictions_and_labels = predictions.select(["prediction", "label_bin"])
    predictions_and_labels.selectExpr("cast(prediction as int) prediction")
    metrics = MulticlassMetrics(predictions_and_labels.rdd.map(tuple))
    evaluator = MulticlassClassificationEvaluator(labelCol = "label_bin", predictionCol = "prediction")

    bin_to_name = {int(row["label_bin"]): row["label_multi"]
                   for row in predictions.select("label_bin", "label_multi").distinct().collect()}

    if binaryClassFlag:                   
        binary_metrics = BinaryClassificationMetrics(predictions_and_labels.select("prediction", "label_bin").rdd.map(tuple))
        areaUnderCurve = binary_metrics.areaUnderROC
    else:
        areaUnderCurve = "na"

    cfsn_temp = metrics.confusionMatrix()
    cfsn_array = cfsn_temp.toArray().astype(int)
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})

    # Build labeled confusion matrix no printing in CSV here anymore, returned as output instead
    ordered_names = [bin_to_name.get(i + 1, f"bin{i+1}") for i in range(cfsn_array.shape[0])]
    cfsn_df = pd.DataFrame(cfsn_array, index=ordered_names, columns=ordered_names)
    cfsn_df.index.name = "Actual"
    cfsn_df.columns.name = "Predicted"

    precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
    recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
    f_measure = evaluator.evaluate(predictions, {evaluator.metricName: "weightedFMeasure"})
    truePositive = evaluator.evaluate(predictions, {evaluator.metricName: "weightedTruePositiveRate"})
    falsePositive = evaluator.evaluate(predictions, {evaluator.metricName: "weightedFalsePositiveRate"})

    total_count = cfsn_array.sum()
    num_classes = cfsn_array.shape[0]
    tp_fp_fn_tn_list = []
    for i in range(num_classes):
        tp = int(cfsn_array[i, i])
        fn = int(cfsn_array[i, :].sum() - tp)
        fp = int(cfsn_array[:, i].sum() - tp)
        tn = int(total_count - tp - fn - fp)
        tactic_name = bin_to_name.get(i + 1, f"bin{i+1}")
        line = f"{tactic_name}: TP={tp} FP={fp} FN={fn} TN={tn}"
        printToLog(f"  {tactic_name} -> TP={tp}, FP={fp}, FN={fn}, TN={tn}", log_location)
        tp_fp_fn_tn_list.append(line)

    distinct_labels = sorted(r["label_bin"] for r in predictions.select("label_bin").distinct().collect())
    for lbl in distinct_labels:
        lbl_p = evaluator.evaluate(predictions, {evaluator.metricName: "precisionByLabel", evaluator.metricLabel: float(lbl)})
        lbl_r = evaluator.evaluate(predictions, {evaluator.metricName: "recallByLabel", evaluator.metricLabel: float(lbl)})
        lbl_f1 = evaluator.evaluate(predictions, {evaluator.metricName: "fMeasureByLabel", evaluator.metricLabel: float(lbl)})
        tactic_name = bin_to_name.get(int(lbl), f"bin{int(lbl)}")
        printToLog(f"  {tactic_name} -> precision={lbl_p:.3f}, recall={lbl_r:.3f}, f1={lbl_f1:.3f}", log_location)

    train_time = (end_randForestTraining - begin_randForestTraining).total_seconds()
    test_time = (end_randForestPredictions - begin_randForestPredictions).total_seconds()

    printToLog("randomForest metrics finished", log_location)

    if countRuns:
        buff = csvAppendBuffer(localNow, conn_server_loc, key, percent_attack_data,
                           len(feature_cols), feature_cols,
                           accuracy, precision, recall, f_measure, areaUnderCurve,
                           truePositive, falsePositive, tp_fp_fn_tn_list,
                           bin_time, train_time, test_time)
                           
        header = csvAppendBuffer(*headerString)
        
        with open(rf_results_location, 'a') as fd:
            if getsize(rf_results_location) == 0:
                fd.write(header)
            fd.write(buff)
            fd.close()

    return cfsn_df   # prints matrix to output instead of CSV
            
def gbtMaster(test, train, binaryClassFlag, bin_time, log_location, gb_results_location, countRuns, localNow, conn_server_loc, key, percent_attack_data, feature_cols):
    gbt = GBTClassifier(featuresCol = "features",
                        labelCol = "label_bin")
                
    begin_gbTraining = datetime.datetime.now()
    gbtModel = gbt.fit(train)
    end_gbTraining = datetime.datetime.now()

    printToLog("GBTClassifier model fit", log_location)
    
    #############
    # Predictions
    #############   
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

    # .astype(int) matches randForestMaster, avoids scientific notation
    gb_cfsn_temp = gbMetrics.confusionMatrix()
    gb_cfsn_array = gb_cfsn_temp.toArray().astype(int)
    gb_cfsn_mtrx = np.array2string(gb_cfsn_array).replace('\n', '')

    gb_accuracy = gbEval.evaluate(gbPredictions, {gbEval.metricName: "accuracy"})

    # weighted-average across ALL classes matches randForestMaster
    gb_precision = gbEval.evaluate(gbPredictions, {gbEval.metricName: "weightedPrecision"})
    gb_recall = gbEval.evaluate(gbPredictions, {gbEval.metricName: "weightedRecall"})
    gb_f_measure = gbEval.evaluate(gbPredictions, {gbEval.metricName: "weightedFMeasure"})
    gb_truePositive = gbEval.evaluate(gbPredictions, {gbEval.metricName: "weightedTruePositiveRate"})
    gb_falsePositive = gbEval.evaluate(gbPredictions, {gbEval.metricName: "weightedFalsePositiveRate"})

    # per-class TP/FP/FN/TN as raw counts matches randForestMaster
    gb_total_count = gb_cfsn_array.sum()
    gb_num_classes = gb_cfsn_array.shape[0]
    gb_tp_fp_fn_tn_list = []
    for i in range(gb_num_classes):
        tp = int(gb_cfsn_array[i, i])
        fn = int(gb_cfsn_array[i, :].sum() - tp)
        fp = int(gb_cfsn_array[:, i].sum() - tp)
        tn = int(gb_total_count - tp - fn - fp)
        line = f"bin{i+1}: TP={tp} FP={fp} FN={fn} TN={tn}"
        printToLog(f"  label_bin {i+1} -> TP={tp}, FP={fp}, FN={fn}, TN={tn}", log_location)
        gb_tp_fp_fn_tn_list.append(line)

    gb_train_time = (end_gbTraining - begin_gbTraining).total_seconds()
    gb_test_time = (end_gbPredictions - begin_gbPredictions).total_seconds()
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
                           gb_tp_fp_fn_tn_list,
                           bin_time,
                           gb_train_time,
                           gb_test_time)
                           
        header = csvAppendBuffer(*headerString)   # added * (was missing here)
        
        with open(gb_results_location, 'a') as fd:
            if getsize(gb_results_location) == 0:
                fd.write(header)
                
            fd.write(gbuff)
            fd.close()
    
    printToLog("GBTClassifier metrics finished", log_location)