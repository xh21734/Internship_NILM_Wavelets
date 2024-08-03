#!/bin/bash
#
#
#SBATCH --job-name=bior4.4_NN_RMS
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=bior4.4/bior4.4_out_RMS_%A.txt
#SBATCH --error=bior4.4/bior4.4_err_RMS_%A.txt
#SBATCH --time=18:00:0
#SBATCH --mem=12G
#SBATCH --account=

# Load in python
module add languages/python/3.12.3

# Go to working dir
cd /user/work/xh21734/Intern/RunModels/

# Load in pre-configured venv
source .venv/bin/activate

# Run the script

python bior4.4_NN.py --idx 1

python bior4.4_NN.py --idx 3

python bior4.4_NN.py --idx 5
