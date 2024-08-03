import time
import numpy as np
from pywt import wavedec
import itertools


Wavelets = ["db1","db2","sym2","sym4","bior4.4"]
Levels = [2,7,12]
combinations = list(itertools.product(Wavelets, Levels))

for job_idx in range(len(combinations)):
    job_params = combinations[job_idx]
    wav = job_params[0]
    lev = job_params[1]
    savpath=wav+" level "+str(lev)
    print(savpath)
    save_path = "/user/work/xh21734/Intern/TestData/DWT/"
    
    plaid_AC_data_total_time_start = time.time()
    plaid_AC_curr_data = np.load("/user/work/xh21734/Intern/TestData/Basic_Run/PLAID_5000Hz_AC_AUGMENTED_data.npy")
    plaid_RMS_data_total_time_start = time.time()
    plaid_RMS_curr_data = np.load("/user/work/xh21734/Intern/TestData/Basic_Run/PLAID_5000Hz_RMS_AUGMENTED_data.npy")
    plaid_AC_labels_total_time_start = time.time()
    plaid_AC_labels = np.load("/user/work/xh21734/Intern/TestData/Basic_Run/PLAID_5000Hz_AC_labels.npy")
    plaid_RMS_labels_total_time_start = time.time()
    plaid_RMS_labels = np.load("/user/work/xh21734/Intern/TestData/Basic_Run/PLAID_5000Hz_RMS_labels.npy")

    plaid_names = ['Hairdryer', 'Vacuum', 'Heater', 'Washing Machine', 'Microwave', 'Fan', 'Incandescent Light Bulb', 'Fridge', 'Air Conditioner', 'Laptop', 'Compact Fluorescent Lamp']

    #Eleven Appliances - Hairdryer, Vacuum, Heater, Washing Machine, Microwave, Fan, Incandescent Light Bulb, Fridge, Air Conditioner, Laptop, Compact Fluorescent Lamp
    n = 11 #eleven appliances

    plaid_AC_labels_eleven_appliances = []
    plaid_RMS_labels_eleven_appliances = []
    plaid_AC_curr_data_eleven_appliances = []
    plaid_RMS_curr_data_eleven_appliances = []

    for i in range(n):
        #Finding indexes of occurences of appliances in datatset
        xindex = np.where(np.isin(plaid_AC_labels,plaid_names[i]))[0].tolist()
        yindex = np.where(np.isin(plaid_RMS_labels,plaid_names[i]))[0].tolist()
        #print(xindex)
        
        plaid_AC_labels_appliance = [plaid_AC_labels[x] for x in xindex]
        plaid_RMS_labels_appliance = [plaid_RMS_labels[y] for y in yindex]
        plaid_AC_curr_data_appliance = [plaid_AC_curr_data[x] for x in xindex]
        plaid_RMS_curr_data_appliance = [plaid_RMS_curr_data[y] for y in yindex]
        
        plaid_AC_labels_eleven_appliances = plaid_AC_labels_eleven_appliances + plaid_AC_labels_appliance
        plaid_RMS_labels_eleven_appliances = plaid_RMS_labels_eleven_appliances + plaid_RMS_labels_appliance
        plaid_AC_curr_data_eleven_appliances = plaid_AC_curr_data_eleven_appliances + plaid_AC_curr_data_appliance
        plaid_RMS_curr_data_eleven_appliances = plaid_RMS_curr_data_eleven_appliances + plaid_RMS_curr_data_appliance

    plaid_AC_dwt_coefs_eleven_appliances = []
    plaid_RMS_dwt_coefs_eleven_appliances = []
    plaid_AC_dwt_data_eleven_appliances = []
    plaid_RMS_dwt_data_eleven_appliances = []

    plaid_AC_app_time_start =[]
    plaid_AC_app_time_end =[]
    plaid_RMS_app_time_start =[]
    plaid_RMS_app_time_end =[]

    for i in range(len(plaid_AC_curr_data_eleven_appliances)):
        plaid_AC_curr_data_eleven_appliances[i].tolist()

        plaid_AC_app_time_start.append(time.time())
        plaid_AC_dwt_coefs = wavedec(plaid_AC_curr_data_eleven_appliances[i], wavelet=wav,level=lev)
        plaid_AC_app_time_end.append(time.time())

        plaid_AC_dwt_coefs_eleven_appliances.append(plaid_AC_dwt_coefs)
        plaid_AC_dwt_data_eleven_appliances.append(np.concatenate(plaid_AC_dwt_coefs_eleven_appliances[i]))


    for j in range(len(plaid_RMS_curr_data_eleven_appliances)):   
        plaid_RMS_curr_data_eleven_appliances[j].tolist()

        plaid_RMS_app_time_start.append(time.time())
        plaid_RMS_dwt_coefs = wavedec(plaid_RMS_curr_data_eleven_appliances[j], wavelet=wav,level=lev)
        plaid_RMS_app_time_end.append(time.time())

        plaid_RMS_dwt_coefs_eleven_appliances.append(plaid_RMS_dwt_coefs)
        plaid_RMS_dwt_data_eleven_appliances.append(np.concatenate(plaid_RMS_dwt_coefs_eleven_appliances[j]))


    print("PLAID data shapes:")
    print(str(len(plaid_AC_dwt_data_eleven_appliances))+"x"+str(len(plaid_AC_dwt_data_eleven_appliances[0])))
    #print("PLAID AC labels shape:")
    print(len(plaid_AC_labels_eleven_appliances))
    #print("PLAID RMS data shape:")
    print(str(len(plaid_RMS_dwt_data_eleven_appliances))+"x"+str(len(plaid_RMS_dwt_data_eleven_appliances[0])))
    #print("PLAID RMS labels shape:")
    print(len(plaid_RMS_labels_eleven_appliances))



    print("PLAID total DWT application time:")

    np.save((save_path  + savpath + "/PLAID_5000Hz_AC_AUGMENTED_data.npy"), np.array(plaid_AC_dwt_data_eleven_appliances))
    plaid_AC_data_total_time_end = time.time()
    print(str(plaid_AC_data_total_time_end-plaid_AC_data_total_time_start))
    np.save((save_path  + savpath + "/PLAID_5000Hz_AC_labels.npy"), np.array(plaid_AC_labels_eleven_appliances))    
    plaid_AC_labels_total_time_end = time.time()
    print(str(plaid_AC_labels_total_time_end-plaid_AC_labels_total_time_start))
    np.save((save_path  + savpath + "/PLAID_5000Hz_RMS_AUGMENTED_data.npy"), np.array(plaid_RMS_dwt_data_eleven_appliances))
    plaid_RMS_data_total_time_end = time.time()
    print(str(plaid_RMS_data_total_time_end-plaid_RMS_data_total_time_start))
    np.save((save_path  + savpath + "/PLAID_5000Hz_RMS_labels.npy"), np.array(plaid_RMS_labels_eleven_appliances))
    plaid_RMS_labels_total_time_end = time.time()
    print(str(plaid_RMS_labels_total_time_end-plaid_RMS_labels_total_time_start))

    print("PLAID average DWT application time: ")
    print(str(np.mean(np.array(plaid_AC_app_time_end) - np.array(plaid_AC_app_time_start))))
    print("N/A")
    print(str(np.mean(np.array(plaid_RMS_app_time_end) - np.array(plaid_RMS_app_time_start))))
    print("N/A")


    whited_AC_data_total_time_start = time.time()
    whited_AC_curr_data = np.load("/user/work/xh21734/Intern/TestData/Basic_Run/WHITED_5000Hz_AC_AUGMENTED_data.npy")
    whited_AC_labels_total_time_start = time.time()
    whited_AC_labels = np.load("/user/work/xh21734/Intern/TestData/Basic_Run/WHITED_5000Hz_AC_labels.npy", allow_pickle=True)
    whited_RMS_data_total_time_start = time.time()
    whited_RMS_curr_data = np.load("/user/work/xh21734/Intern/TestData/Basic_Run/WHITED_5000Hz_RMS_AUGMENTED_data.npy")
    whited_RMS_labels_total_time_start = time.time()
    whited_RMS_labels = np.load("/user/work/xh21734/Intern/TestData/Basic_Run/WHITED_5000Hz_RMS_labels.npy")

    whited_AC_labels = np.concatenate(whited_AC_labels, axis=0)

    whited_names = ['HairDryer', 'VacuumCleaner', 'FanHeater', 'WashingMachine', 'Microwave', 'Fan', 'LightBulb', 'Fridge', 'AC', 'Laptop', 'CFL']

    #Eleven Appliances - Hairdryer, Vacuum, Heater, Washing Machine, Microwave, Fan, Incandescent Light Bulb, Fridge, Air Conditioner, Laptop, Compact Fluorescent Lamp
    n = 11 #eleven appliances

    whited_AC_labels_eleven_appliances = []
    whited_RMS_labels_eleven_appliances = []
    whited_AC_curr_data_eleven_appliances = []
    whited_RMS_curr_data_eleven_appliances = []

    for i in range(n):
        aindex = np.where(np.isin(whited_AC_labels,whited_names[i]))[0].tolist()
        bindex = np.where(np.isin(whited_RMS_labels,whited_names[i]))[0].tolist()
        
        whited_AC_labels_appliance = [whited_AC_labels[a] for a in aindex]
        whited_RMS_labels_appliance = [whited_RMS_labels[b] for b in bindex]
        whited_AC_curr_data_appliance = [whited_AC_curr_data[a] for a in aindex]
        whited_RMS_curr_data_appliance = [whited_RMS_curr_data[b] for b in bindex]
        
        whited_AC_labels_eleven_appliances = whited_AC_labels_eleven_appliances + whited_AC_labels_appliance
        whited_RMS_labels_eleven_appliances = whited_RMS_labels_eleven_appliances + whited_RMS_labels_appliance
        whited_AC_curr_data_eleven_appliances = whited_AC_curr_data_eleven_appliances + whited_AC_curr_data_appliance
        whited_RMS_curr_data_eleven_appliances = whited_RMS_curr_data_eleven_appliances + whited_RMS_curr_data_appliance

    whited_AC_dwt_coefs_eleven_appliances = []
    whited_RMS_dwt_coefs_eleven_appliances = []
    whited_AC_dwt_data_eleven_appliances = []
    whited_RMS_dwt_data_eleven_appliances = []

    whited_AC_app_time_start =[]
    whited_AC_app_time_end =[]
    whited_RMS_app_time_start =[]
    whited_RMS_app_time_end =[]

    for i in range(len(whited_AC_curr_data_eleven_appliances)):
        whited_AC_curr_data_eleven_appliances[i].tolist()

        whited_AC_app_time_start.append(time.time())
        whited_AC_dwt_coefs = wavedec(whited_AC_curr_data_eleven_appliances[i], wavelet=wav,level=lev)
        whited_AC_app_time_end.append(time.time())

        whited_AC_dwt_coefs_eleven_appliances.append(whited_AC_dwt_coefs)
        whited_AC_dwt_data_eleven_appliances.append(np.concatenate(whited_AC_dwt_coefs_eleven_appliances[i]))

    for j in range(len(whited_RMS_curr_data_eleven_appliances)):
        whited_RMS_curr_data_eleven_appliances[j].tolist()

        whited_RMS_app_time_start.append(time.time())
        whited_RMS_dwt_coefs = wavedec(whited_RMS_curr_data_eleven_appliances[j], wavelet=wav,level=lev)
        whited_RMS_app_time_end.append(time.time())

        whited_RMS_dwt_coefs_eleven_appliances.append(whited_RMS_dwt_coefs)
        whited_RMS_dwt_data_eleven_appliances.append(np.concatenate(whited_RMS_dwt_coefs_eleven_appliances[j]))


        
    print("WHITED data shapes:")
    print(str(len(whited_AC_dwt_data_eleven_appliances))+"x"+str(len(whited_AC_dwt_data_eleven_appliances[0])))
    #print("WHITED AC labels shape:")
    print(len(whited_AC_labels_eleven_appliances))
    #print("WHITED RMS data shape:")
    print(str(len(whited_RMS_dwt_data_eleven_appliances))+"x"+str(len(whited_RMS_dwt_data_eleven_appliances[0])))
    #print("WHITED RMS labels shape:")
    print(len(whited_RMS_labels_eleven_appliances))



    print("WHITED total DWT application time:")

    np.save((save_path  + savpath + "/WHITED_5000Hz_AC_AUGMENTED_data.npy"), np.array(whited_AC_dwt_data_eleven_appliances))
    whited_AC_data_total_time_end = time.time()
    print(str(whited_AC_data_total_time_end-whited_AC_data_total_time_start))
    np.save((save_path  + savpath + "/WHITED_5000Hz_AC_labels.npy"), np.array(whited_AC_labels_eleven_appliances))    
    whited_AC_labels_total_time_end = time.time()
    print(str(whited_AC_labels_total_time_end-whited_AC_labels_total_time_start))
    np.save((save_path  + savpath + "/WHITED_5000Hz_RMS_AUGMENTED_data.npy"), np.array(whited_RMS_dwt_data_eleven_appliances))
    whited_RMS_data_total_time_end = time.time()
    print(str(whited_RMS_data_total_time_end-whited_RMS_data_total_time_start))
    np.save((save_path  + savpath + "/WHITED_5000Hz_RMS_labels.npy"), np.array(whited_RMS_labels_eleven_appliances))
    whited_RMS_labels_total_time_end = time.time()
    print(str(whited_RMS_labels_total_time_end-whited_RMS_labels_total_time_start))

    print("WHITED average DWT application time: ")
    print(str(np.mean(np.array(whited_AC_app_time_end) - np.array(whited_AC_app_time_start))))
    print("N/A")
    print(str(np.mean(np.array(whited_RMS_app_time_end) - np.array(whited_RMS_app_time_start))))
    print("N/A")
    