#!/bin/bash
#SBATCH --time=7:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000

module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0
pip install pycuda --user

source $HOME/venvs/thesis_env/bin/activate

MODULE_NAME=${MODULE_NAME:-"pan_organized/sequence_labeling/seqlabel_experiment_runner.py"}

#ADDITIONAL_PYTHONPATH=${ADDITIONAL_PYTHONPATH:-"."}
export PYTHONPATH=$HOME/pan_organized:$PYTHONPATH
echo "PYTHONPATH is: $PYTHONPATH"

#export PYTHONPATH=$PYTHONPATH:$ADDITIONAL_PYTHONPATH

for lang in en; do
    label="seqlabel_experiment_-${lang}"
    python $MODULE_NAME $lang 5 \
        --test 0 \
        --experim_label $label \
        --pause_after_fold 0 \
        --pause_after_model 0 \
        --max_seq_length 256 \
        --gpu 0
done

deactivate