#!/bin/bash
#
#
#SBATCH --job-name=sym4_NN_AC
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=sym4/sym4_out_AC_%A.txt
#SBATCH --error=sym4/sym4_err_AC_%A.txt
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
python sym4_NN.py --idx 0

python sym4_NN.py --idx 2

python sym4_NN.py --idx 4