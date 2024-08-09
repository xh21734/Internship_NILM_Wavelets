import numpy as np
import pandas as pd
import glob
import scipy
import time
import re
import os
import json
import itertools

Wavelets = ["db1","db2","sym2","sym4","bior4.4"]
Levels = [2,7,12]
combinations = list(itertools.product(Wavelets, Levels))
sheets = []

for job_idx in range(len(combinations)):
    job_params = combinations[job_idx]
    wav = job_params[0]
    lev = job_params[1]
    sheets.append(wav+" level "+str(lev))

sheets.append("Basic_Run")
sheets.append("CWT")
for i in range(len(sheets)):
    sheet = sheets[i]
    print(sheet)
    main_dir = "/user/work/xh21734/Intern/Reports/DWT/"  + sheet +"/"
    save_dir = "/user/work/xh21734/Intern/Results/DWT/"
    sub_dir = ["AC_plaid_reports", "AC_whited_reports", "AC_plaid_and_whited_reports","RMS_plaid_reports", "RMS_whited_reports", "RMS_plaid_and_whited_reports"]


    #columns : ["dataset", "input_type", "algorithm", "sample_size", "frequency",
    #             "precision", "recall", "F1-score", "compute_time(sec)"]
    temp_metric_list = []
    num_classes = 11
    for folder in sub_dir:
        input_type = folder.split("_")[0]
        dataset = "_".join(("_".join(folder.split("_")[1:])).split("_")[:-1])
        print(main_dir+folder+"/*.json")
        for filename in glob.glob((main_dir+folder+"/*.json")):
            print(filename)
            algorithm = filename.split("/")[9].split("_")[0]
            sample_size = int(filename.split("/")[9].split("_")[1].split("samples")[0])
            freq = int(filename.split("/")[9].split("_")[2].split("Hz")[0])
            #print(algorithm, sample_size, freq)
            with open(filename, "r") as readfile:
                json_data = json.load(readfile)
            precision = json_data["weighted avg"]["precision"]
            recall = json_data["weighted avg"]["recall"]
            f1_score = json_data["weighted avg"]["f1-score"]

            #getting compute time as well
            txt_log_base = "_log.txt"
            samples_per_class = int(sample_size/num_classes)
            txt_log_size = txt_log_base
            if algorithm in ["knn", "svm", "xgboost"]:
                txt_log_name = "benchmark" + txt_log_size
                model_file = filename[:-11] + "model.pkl"
            else:
                txt_log_name = "NN" + txt_log_size
                model_file = filename[:-11] + "model.keras"

            #using glob to find all files
            seconds = np.nan
            all_lines = []
            for filename2 in glob.glob((main_dir+folder+"/*"+txt_log_name)):
                #print(filename)
                with open(filename2) as text_file:
                    file_lines = text_file.readlines()
                all_lines.extend(file_lines)
            
            myregex = algorithm + " " + str(samples_per_class) + " sample size at " + str(freq) + " Hz :"
            #print(myregex)
            #print(line)
            for line in all_lines:
                #print(myregex)
                
                #print(line)
                flag = re.findall(myregex, line, re.IGNORECASE)
                if flag:
                        seconds = round(float(line.split(":")[1].split(" seconds")[0]), 2)
                        #print(seconds, line)
                        break
                
            #getting file size as well
            file_size = int(os.path.getsize(model_file)) / 1_000_000 #converting to MB
            #print(file_size)

            file_meta = [dataset, input_type, algorithm, sample_size, 
                        freq, precision, recall, f1_score, seconds, file_size]
            temp_metric_list.append(file_meta)
    metrics_df = pd.DataFrame(temp_metric_list, columns=["dataset", "input_type", "algorithm", 
                                                            "sample_size", "frequency",
                                                            "precision", "recall", "F1-score", 
                                                            "compute_time(sec)", "file_size(MB)"])



    #finding best metrics for each algorithm on each dataset
    grouped = metrics_df.loc[metrics_df.groupby(["input_type","dataset", "algorithm"])["F1-score"].idxmax()]\
                            [["dataset", "input_type", "algorithm", "sample_size", "frequency", "F1-score","compute_time(sec)", "file_size(MB)"]]
                            
    metrics_df.to_csv(save_dir + sheet + ' metrics.csv')
    grouped.to_csv(save_dir + sheet + ' grouped.csv')