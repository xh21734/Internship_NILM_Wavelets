import os
import itertools

save_path = "/user/work/xh21734/Intern/TestData/Basic_Run/"
os.makedirs(save_path)

save_path = "/user/work/xh21734/Intern/TestData/CWT/"
os.makedirs(save_path)


Wavelets = ["db1","db2","sym2","sym4","bior4.4"]
Levels = [2,7,12]
combinations = list(itertools.product(Wavelets, Levels))

save_path = "/user/work/xh21734/Intern/TestData/DWT/"
os.makedirs(save_path)

for job_idx in range(len(combinations)):
    job_params = combinations[job_idx]
    wav = job_params[0]
    lev = job_params[1]
    savpath=wav+" level "+str(lev)
    print(savpath)
    save_path = "/user/work/xh21734/Intern/TestData/DWT/" + savpath
    
    if not os.path.exists(savpath):
        os.makedirs(save_path)