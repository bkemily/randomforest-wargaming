# Apache Spark imports
import pyspark
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import split, col, isnan, when, count

# Standard Python imports
import traceback
import sys
import numpy as np
import datetime


def rename_columns(df_in):
    # Dictionary to store old:new column names.
    #                 Key:                Value
    #                 ---------------------------------
    #                 Original Col Name:  New Col Name
    col_names_dict = {"resp_pkts":        "resp_pkts",
                  "label_tactic":     "label_multi",
                  "service":          "service",
                  "orig_ip_bytes":    "orig_ip_bytes",
                  "local_resp":       "local_resp",
                  "missed_bytes":     "missed_bytes",
                  "proto":            "protocol",
                  "duration":         "duration",
                  "conn_state":       "conn_state",
                  "dest_ip_zeek":     "dest_ip",
                  "orig_pkts":        "orig_pkts",
                  "community_id":     "community_id",
                  "resp_ip_bytes":    "resp_ip_bytes",
                  "dest_port_zeek":   "dest_port",
                  "orig_bytes":       "orig_bytes",
                  "local_orig":       "local_orig",
                  "datetime":         "datetime",
                  "history":          "history",
                  "resp_bytes":       "resp_bytes",
                  "uid":              "uid",
                  "src_port_zeek":    "src_port",
                  "ts":               "ts",
                  "src_ip_zeek":      "src_ip"}


    # Function that renames the columns provided in col_names_dict.
    for c in col_names_dict:
        if c not in df_in.columns:
            raise Exception("Cannot rename column " + str(c) + " to " + str(col_names_dict[c]) + \
                            ".\n\t" + str(c) + " is not name of column in dataframe.\n")
        df_in = df_in.withColumnRenamed(c, col_names_dict[c])
    return df_in


'''
    This block generates a python dictionary {key: value} of {mitre tactic: dataframe}.  
'''

def create_dict_of_dfs (df, benign_label, label_col_name, percent_attack_data, num_top_labels, logFlag=False, log_location=""): 
    # User defined variables.  
    # benign_label is the name used to identify benign data in the dataframes.  
    # label_col_name is the column name for the label to filter.
    # percent_attack_data is the desired percentage of attack data relative to benign.
    # num_top_labels is the number of labels you want to pull from the original dataset.
    # by default, this script sorts the labels by count in descending order.
    #benign_label = "none"
    #label_col_name = "label_multi"
    #percent_attack_data = 0.3
    #num_top_labels = 3


    # Function that calculates the number of benign records required to meet a defined attack
    # data percentage.  Requires user defined attack data percentage, and the number of attack
    # records in the dataframe.  
    def benign_data_count(attack_count, attack_pct):
        if attack_pct <= 0.0 or attack_pct > 1.0:
            raise Exception("Invalid Percent for Attack Records 0.0 < Attack Percent <= 1.0")
        return(((1.0 - attack_pct) * attack_count) / attack_pct)


    # Generates a dataframe that contains the counts of labels
    #df_unique_counts = df.groupBy(label_col_name).count().orderBy(col(label_col_name).desc()) #sorts alphabetically
    df_unique_counts = df.groupBy(label_col_name).count().orderBy(col("count").desc()) #sorts by counts
    #df_unique_counts.show()

    # There's no guarantee that the benign label will be one of the most populous, so we are
    # going to pull that out of the unique label counts dataframe.  
    df_unique_counts = df_unique_counts.filter(col(label_col_name) != benign_label)

    # Next, a new dataframe with the benign (non-attack) records only. 
    df_benign = df.filter(col(label_col_name) == benign_label)

    # Next, select the top n records returned, where n is the user defined value num_top_labels. 
    # df.head() returns a list of pyspark.sql.Row, so accessing the data is a bit different.
    unique_non_benign = df_unique_counts.head(num_top_labels)

    # One line for loop using dictionary comprehension.  Dictionary key is attack id (ex: T1595),
    # and value is filtered dataframe containing all rows with key (ex: T1595) as its label.  
    dict_of_dfs = {row[0]: df.filter(col(label_col_name) == row[0]) for row in unique_non_benign}

    # for loop to add user defined number of records to meet user defined attack data percentage.  
    for df_label in dict_of_dfs:
        # Use benign_data_count function to determine the number of benign records needed to 
        # achieve defined percentage of attack data.  
        benign_count = benign_data_count(dict_of_dfs[df_label].count(), percent_attack_data)
    
        if logFlag == True:
            printToLog("" + str(df_label) + " found in dataset with " + \
               str(dict_of_dfs[df_label].count()) + " unique records", log_location)
    
        # Get random sample of benign data.  It is possible that there are not enough benign records to
        # completely fulfill the percent attack data requirement.  Thus, modulo is used to get the 
        # remainder left after removing n full benign dataframes from the desired benign records.  
        # Then, a for loop is used to make full copies of the benign dataframe to fill in the remaining
        # full benign dataframes required.
        df_tmp = df_benign.sample(False, ((benign_count % df_benign.count()) / df_benign.count()), 1234)  
    
        # This for loop to fills in the remaining full benign dataframes required to obtain desired 
        # attack data percentage.  
        # Notes:  in python // performs integer division.  If int(benign_count)//int(df_benign.count()) == 0,
        # then the for loop does not execute.  Thus, no full benign dataframe copies are made.
        for i in range(0, (int(benign_count)//int(df_benign.count()))):
            df_tmp = df_tmp.unionByName(df_benign) # Use unionByName to copy full dataframes to df_tmp
    
        # Use unionByName to copy benign data to the attack dataframes in the dictionary. 
        dict_of_dfs[df_label] = dict_of_dfs[df_label].unionByName(df_tmp)

    return dict_of_dfs

def print_df_counts(dict_of_dfs, label_col_name):
    # After the dictionary has been populated, we can perform a sanity check to verify that the record counts 
    # match the desired percentages.  Note that for loops using dictionaries iterates over the key value 
    # (i.e. "T1595", "T1046", etc.)
    for key in dict_of_dfs:
        print(key + " Total Records:  " + str(dict_of_dfs[key].count()))
        dict_of_dfs[key].groupBy(label_col_name).count().show()
        print()

def printToLog(addMe, logLocation):
    with open(logLocation, 'a') as fd:
        fd.write("" + str(datetime.datetime.now()) + ": " + str(addMe) + "\n")
