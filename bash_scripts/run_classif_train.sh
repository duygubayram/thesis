#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000

module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0
pip install pycuda --user

source $HOME/venvs/thesis_env/bin/activate

python pan_organized/classif_experim/classif_experiment_runner.py

deactivate