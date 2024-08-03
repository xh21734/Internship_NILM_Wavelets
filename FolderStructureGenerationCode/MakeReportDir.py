import os
import itertools

save_path = "/user/work/xh21734/Intern/TestReports/Basic_Run/"
os.makedirs(save_path)
os.makedirs(save_path+"/AC_plaid_and_whited_reports/")
os.makedirs(save_path+"/AC_plaid_reports/")
os.makedirs(save_path+"/AC_whited_reports/")
os.makedirs(save_path+"/RMS_plaid_and_whited_reports/")
os.makedirs(save_path+"/RMS_plaid_reports/")
os.makedirs(save_path+"/RMS_whited_reports/")


save_path = "/user/work/xh21734/Intern/TestReports/CWT/"
os.makedirs(save_path)
os.makedirs(save_path+"/AC_plaid_and_whited_reports/")
os.makedirs(save_path+"/AC_plaid_reports/")
os.makedirs(save_path+"/AC_whited_reports/")
os.makedirs(save_path+"/RMS_plaid_and_whited_reports/")
os.makedirs(save_path+"/RMS_plaid_reports/")
os.makedirs(save_path+"/RMS_whited_reports/")



Wavelets = ["db1","db2","sym2","sym4","bior4.4"]
Levels = [2,7,12]
combinations = list(itertools.product(Wavelets, Levels))

save_path = "/user/work/xh21734/Intern/TestReports/DWT/"
os.makedirs(save_path)

for job_idx in range(len(combinations)):
    job_params = combinations[job_idx]
    wav = job_params[0]
    lev = job_params[1]
    savpath=wav+" level "+str(lev)
    print(savpath)
    save_path = "/user/work/xh21734/Intern/TestReports/DWT/" + savpath
    
    if not os.path.exists(savpath):
        os.makedirs(save_path)
        os.makedirs(save_path+"/AC_plaid_and_whited_reports/")
        os.makedirs(save_path+"/AC_plaid_reports/")
        os.makedirs(save_path+"/AC_whited_reports/")
        os.makedirs(save_path+"/RMS_plaid_and_whited_reports/")
        os.makedirs(save_path+"/RMS_plaid_reports/")
        os.makedirs(save_path+"/RMS_whited_reports/")