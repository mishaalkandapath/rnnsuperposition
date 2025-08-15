#!/bin/bash

FILENAME=$1

cd $SLURM_TMPDIR
module load python/3.10
virtualenv env -p python3.10
source env/bin/activate

pip install -r /home/mishaalk/scratch/requirements.txt
deactivate

cp -r /home/mishaalk/projects/def-gpenn/mishaalk/rnnsuperposition/ .
cp -r /home/mishaalk/scratch/$FILENAME .