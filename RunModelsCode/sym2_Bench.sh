#!/bin/bash
#
#
#SBATCH --job-name=sym2_Bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=sym2/sym2_test_out_Bench_%A.txt
#SBATCH --error=sym2/sym2_test_err_Bench_%A.txt
#SBATCH --time=04:00:0
#SBATCH --mem=12G
#SBATCH --account=

# Load in python
module add languages/python/3.12.3

# Go to working dir
cd /user/work/xh21734/Intern/RunModels/

# Load in pre-configured venv
source .venv/bin/activate

# Run the script
python sym2_Bench.py --idx 0

python sym2_Bench.py --idx 1

python sym2_Bench.py --idx 2

python sym2_Bench.py --idx 3

python sym2_Bench.py --idx 4

python sym2_Bench.py --idx 5
