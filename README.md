# thesis

Slightly modified from the original: https://github.com/dkorenci/pan-clef-2024-oppositional/tree/main

#### To run on Habrok cluster:
create virtual environment

```python3 -m venv $HOME/venvs/first_env```

activate

'''source $HOME/venvs/first_env/bin/activate'''

install requirements

#### Finetune classifier (saves the best fold):
'''sbatch thesis/bash_scripts/run_classif_train.sh'''

#### Evaluate classifier:
'''sbatch thesis/bash_scripts/run_classif_eval.sh'''
