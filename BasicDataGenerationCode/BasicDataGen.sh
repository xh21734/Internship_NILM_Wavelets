#!/bin/bash
#
#
#SBATCH --job-name=BasicDataGen
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=Output/BasicDataGen%A_%a.txt
#SBATCH --error=Error/BasicDataGen%A_%a.txt
#SBATCH --time=01:00:0
#SBATCH --mem=12G
#SBATCH --account=

# Load in python
module add languages/python/3.12.3

# Go to working dir
cd /user/work/xh21734/Intern/DataGenerationCode/

# Load in pre-configured venv
source .venv/bin/activate

# Run the script
python BasicDataGen.py
