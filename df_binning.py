from cmath import nan
import pyspark
from pyspark.sql import functions as f
from pyspark.sql.window import Window
import math as m
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import StringIndexer
import builtins
import sys
import datetime

from  generate_individual_labels_dataframe import *


def generate_df_dict(isBinaryLabelTrue, df, benign_label, label_col_name, percent_attack_data, num_top_labels):
    df_renamed = rename_columns(df)

    if (isBinaryLabelTrue):
        entire_dict_of_dfs = create_dict_of_dfs (df_renamed, benign_label, label_col_name, percent_attack_data, num_top_labels)
        #for now, perform ML algo on these 2 label techniques, the others have too few attack instances
        #in those cases, data binning may remove attack instaces altogether and error out
        dict_of_dfs = {label_tech:entire_dict_of_dfs[label_tech] for label_tech in ("Reconnaissance", "Discovery")}
    else:
        dict_of_dfs = { "All label techniques" : df_renamed }


    print_df_counts(dict_of_dfs, "label_multi")

    return (dict_of_dfs)


# This method results in a trimmed version of the specified
# dataframe column being returned. Nonapplicable values are dropped in 
# a new single column df, after which the percentTrim is applied to 
# both ends of the new df's single column, such that the top and bottom
# percent (equal to the percentTrim) are removed.

def returnTrimmedDF(data_frame, col_name, percentTrim):
    new_df = data_frame.select(col_name).dropna()
    new_df = new_df.sort(col_name)

    count = new_df.count()
    outlier = m.floor(count * percentTrim)
    
    w = Window.orderBy(f.monotonically_increasing_id())
    new_df = new_df.withColumn("trim_row_num", f.row_number().over(w))

    new_df = new_df.filter(f.col("trim_row_num") >= outlier)
    new_df = new_df.filter(f.col("trim_row_num") <= (count - outlier))
    
    return new_df.select(col_name)

# PySpark DF version of the method to generate a tuple of edges for bins

def genNumericEdges(clean_df, col_name):
    calc_df = clean_df.select(f.mean(col_name),f.stddev(col_name), f.min(col_name)).collect()
    mean_val = calc_df[0][0]
    stddev_val = calc_df[0][1]
    min_val = calc_df[0][2]

    if stddev_val == f.nanvl:
        stddev_val = 0

    if mean_val == f.nanvl:
        mean_val = 0

    # This logic was inserted to avoid having multiple bins within 
    # negative ranges if the min value did not go below 0.
    
    if min_val >= 0:
        while mean_val - 2* stddev_val < 0:
            mean_val += stddev_val

    edge0 = float('-inf')
    edge1 = mean_val - stddev_val * 2
    edge2 = mean_val - stddev_val
    edge3 = mean_val
    edge4 = mean_val + stddev_val
    edge5 = mean_val + stddev_val * 2
    edge6 = float('inf')
    edges = [edge0, edge1, edge2, edge3, edge4, edge5, edge6]
    edges_distinct = []

    for i in edges:
        if edges_distinct.__contains__(i):
            continue
        else: edges_distinct = edges_distinct + [i]
    
    return edges_distinct

#Pyspark use generated edges to bin using the Bucketizer functionality. Set null values to -1, then bump everything up by +1

def binNumericFeature(df, col_name,new_col_name, edges, replace_bool):
    
    df_binned = Bucketizer(splits = edges,  inputCol = col_name, outputCol = new_col_name).setHandleInvalid("keep").transform(df)
    
    df_binned = df_binned.withColumn(new_col_name, f.col(new_col_name).cast('int')).fillna(-1, subset = [new_col_name])
    
    df_binned= df_binned.withColumn(new_col_name,df_binned[new_col_name] + 1)

    #Should be triggered on boolean

    if replace_bool == True:
        df_binned = df_binned.drop(col_name).withColumnRenamed(new_col_name, col_name)
    
    return df_binned


#Combined the above methods to output a new df with the binned column.

def genNumericBinnedDF(df, col_name, trim, replace_bool):
    
    new_col_name = col_name + "_bin"

    # Create a sorted and trimmed version of the col_df to 
    # use for edge creation
    
    trimmed_df = returnTrimmedDF(df.select(col_name), col_name, trim)
    
    # Use the trimmed_df to generate edges based on standard
    # deviation schema. Create an edge range to alter final bin values
    
    edge_bins = genNumericEdges(trimmed_df, col_name)

    # Use list of edges generated to bin column using Bucketizer
    
    binned_df = binNumericFeature(df,col_name, new_col_name, edge_bins, replace_bool)

    return binned_df

#Assigns the IP class designation (A, B, C, D, E) as 1, 2, 3, 4, and 5.
#Null and non applicable values get assigned a value of 0. Everything else, 6.

@f.udf
def ipClass(string_val):
    bin_num = 0
    
    if  string_val == f.nanvl or string_val.find(".") == -1:
        return bin_num
    
    first = int(string_val.split('.')[0])

    # Class A
    if first >= 0 and first <= 127:
        bin_num = 1

    # Class B
    elif first <= 191:
        bin_num = 2
    
    # Class C
    elif first <= 223:
        bin_num = 3

    # Class D
    elif first <= 239:
        bin_num = 4

    # Class E
    elif first <= 254:
        bin_num = 5
    
    else: bin_num = 6

    return bin_num

#Using the udf method on the dataframe and column read in

def genIPBinnedDF(df, col_name, replace_bool):
    new_col_name = col_name + "_bin"
    new_df = df.withColumn(new_col_name, ipClass(col_name).cast("int"))
    
    if replace_bool == True:
        new_df = new_df.drop(col_name).withColumnRenamed(new_col_name, col_name)
    
    return new_df

# UDF method for binning port codes, based on Dr. Minks 
# recomendation. Bins for the well known, registered, and 
# dynamic/private port ranges.

@f.udf
def portClass(string_val):
    bin_num = 0
    
    if  string_val == f.nanvl:
        return bin_num

    port_val = int(string_val)

    #Well known ports
    if port_val >= 0 and port_val <= 1023:
        bin_num = 1

    #Registered ports
    elif port_val <= 49151:
        bin_num = 2
    
    #Dynamic/Private ports
    elif port_val <= 65535:
        bin_num = 3

    #Other
    else: bin_num = 4

    return bin_num

#Using the udf method on the dataframe and port column that was read in:

def genPortBinnedDF(df, col_name, replace_bool):
    new_col_name = col_name + "_bin"
    new_df = df.withColumn(new_col_name, portClass(col_name).cast("int"))
    
    if replace_bool == True:
        new_df = new_df.drop(col_name).withColumnRenamed(new_col_name, col_name)
    
    return new_df

# This method just bins using the string indexer

def genStringIndexBinnedDF(df, col_name, replace_bool):
    
    df = df.withColumn(col_name, df.__getattr__(col_name).cast("string"))
    bin_col_name = col_name + '_bin'

    df = StringIndexer(inputCol=col_name, outputCol= bin_col_name, handleInvalid='keep') \
        .fit(df).transform(df)
        
    df = df.withColumn(bin_col_name,  df.__getattr__(bin_col_name).cast('integer'))
    
    if replace_bool == True:
        df = df.drop(col_name).withColumnRenamed(bin_col_name, col_name)

    return df

# This method is pretty self contained, generating a df with the counts of each occurring value,
# sorting by occurrence, then aggregating the sum until the percent_aggr (%) of the total occurences are 
# covered. These values will be assigned their own bins, while everything else is pooled into a single bin.

def genNominalBinnedDF(df, col_name, percent_aggr, replace_bool):

    bin_col_name = col_name + '_bin'
    col_ref = col_name + '_ref'
    
    w = Window.orderBy(f.monotonically_increasing_id())
    
    df_bin = df.select(col_name).groupBy(col_name).count().sort("count", ascending=False).dropna()
    df_bin = df_bin.withColumn(bin_col_name, f.row_number().over(w)).withColumnRenamed(col_name, col_ref)
    
    df_bin.persist()

    unique_count = df_bin.count()
    sum_total = df_bin.select("count").groupBy().sum().collect()[0][0]
    sum_aggr = 0
    j = 0
    
    for i in range(unique_count):
        sum_aggr += df_bin.collect()[i][1]
        
        if sum_aggr <= 0:
            break
        elif sum_aggr >= (sum_total * percent_aggr):
            j = i
            break
        else:
            continue
    
    j = builtins.max(j+1, builtins.min(5, unique_count))
    
    # gives min() takes 1 positional argument but 2 were given error
    # error fixed: import conflict from pyspark.sql.functions was causing 
    # this (https://stackoverflow.com/questions/36604460/python-function-such-as-max-doesnt-work-in-pyspark-application)
   
    df_bin = df_bin.filter( f.col(bin_col_name) <= j ).drop("count")
    
    df = df.join(df_bin, df.__getattr__(col_name) == df_bin.__getattr__(col_ref),"left").drop(col_ref)
    
    df_bin.unpersist()

    #Where the original column was null, assigned a 0 value
    
    df = df.withColumn(bin_col_name, f.when(df[col_name].isNull(), 0).otherwise(df[bin_col_name] )) 
    
    #Where the original column was not null, but not included in the top 80% select, assign +1 to the highest
    #current value

    df = df.withColumn(bin_col_name, f.when(df[bin_col_name].isNull(), j+1).otherwise(df[bin_col_name] )) 

    #If the replace boolean is true, replace the original column with the numeric bin column

    if replace_bool == True:
        df = df.drop(col_name).withColumnRenamed(bin_col_name, col_name)

    return df

#In this part I used the above methods for binning individual columns, then bring them all together. The result is the 
#indexed_binned_df data frame, with all attributes binned with integers, for use in our ML aglorithms.

def genFullBinnedDF (df, attrList_ip_addr, attrList_port, attrList_boolean, attrList_nominal, nominal_percent_agg, attrList_numeric, numeric_percent_trim, replace_bool):
    
    for i in attrList_ip_addr:
        df = genIPBinnedDF(df, i, replace_bool)

    for i in attrList_port:
        df = genPortBinnedDF(df, i, replace_bool)

    for i in attrList_boolean:
        df = genStringIndexBinnedDF(df, i, replace_bool)

    for i in attrList_nominal:
        df = genStringIndexBinnedDF(df, i, replace_bool)
        #df = genNominalBinnedDF(df, i, nominal_percent_agg, replace_bool)

    for i in attrList_numeric:
        df = genNumericBinnedDF(df, i, numeric_percent_trim, replace_bool)

    return df