# Internship_NILM_Wavelets

This repo hopefully contains all required code to create folder structre, generate original augmented datasests (from TLY), applying wavelet transforms, create and test ML models and finally obtain all results!

The first thing to do is setup the file structure on your WinSCP HPC directory (on the "work" side - more storage). I reccommend the following folders:
- A "data" folder which will contain the PLAID & WHITED files, Original Datasets, Augmented Datasets, CWT Applied Datasests and DWT Applied Datasets - this can be setup with the MakeDataDir.py file
- A "results" folder  - this can be setup with the MakeDataDir.py file too (change save path)
- A "reports" folder - this can be setup with the MakeReportsDir.py file
- A "DataGenerationCode" folder - this should contain "Output", "Error" and "Superseded" folders ("Superseded" folders inside the "Output" and "Error" folders too)
- A "RunModels" folder - this should contain "Output", "Error" and "Superseded" folders ("Superseded" folders inside the "Output" and "Error" folders too)

Next download the PLAID and WHITED datasets that can be found here (they are large so require a little time to download and move over to the HPC directories):
https://figshare.com/articles/dataset/PLAID_2017/11605215?file=21003861
https://www.cs.cit.tum.de/dis/resources/whited/

Next download all my code and place in relevant folders in the HPC directories (along with all .sh and requirements.txt files).

Use the PUTTY or BitVise client to log onto the HPC and locate to the "DataGenerationCode" and "RunModels directories individually. To set these up do the following in the client:
1.  module add languages/python/3.12.3
2.  python -m venv .venv
3.  source .venv/bin/activate
4.  pip install -r requirements.txt

Now you are ready to run some jobs (make sure to add your account number to all .sh files)!

Run in the following order and use how you wish(making sure to change directories where necessary!)
1.  "BasicDataGenerationCode"
2.  "ApplyingWaveletsCode"
3.  "RunModelsCode"
4.  "GetResultsCode"

This will eventually give all results in two excel files (metrics and grouped) and you can then present how you wish!

Hope you enjoy and let me know of any issues :)
