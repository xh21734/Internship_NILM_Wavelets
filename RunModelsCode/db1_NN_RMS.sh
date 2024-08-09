#!/bin/bash
#
#
#SBATCH --job-name=db1_NN_RMS
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=db1/db1_out_RMS_%A.txt
#SBATCH --error=db1/db1_err_RMS_%A.txt
#SBATCH --time=24:00:0
#SBATCH --mem=12G
#SBATCH --account=mech029804

# Load in python
module add languages/python/3.12.3

# Go to working dir
cd /user/work/xh21734/Intern/RunModels/

# Load in pre-configured venv
source .venv/bin/activate

export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib

# Run the script

python db1_NN.py --idx 1

python db1_NN.py --idx 3

python db1_NN.py --idx 5
