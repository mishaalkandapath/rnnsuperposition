#!/bin/bash

FILENAME=$1

cd $SLURM_TMPDIR
module load python/3.10
virtualenv env -p python3.10
source env/bin/activate

pip install -r /home/mishaalk/scratch/requirements.txt
deactivate

mkdir data
# cp /home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/data/dataset_splits.pkl data/dataset_splits.pkl
# cp /home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/data/input_sentences.csv data/input_sentences.csv

cp /home/mishaalk/scratch/$FILENAME $FILENAME
# if [ "$FILENAME" == "simplestlambda.tgz" ]; then
#     cp /home/mishaalk/scratch/data_original_cedar.tgz data_original_cedar.tgz
#     tar -zxf data_original_cedar.tgz  --strip-components=1 -C data/ data/
#     tar -zxf $FILENAME --strip-components=3 -C data/ simplestlambda/lambdaBERT/data/
#     rm -rf data_original_cedar.tgz
# else
#     tar -zxf $FILENAME --strip-components=1 -C data/ data/
# fi
# cp /home/mishaalk/scratch/data_original_cedar.tgz data_original_cedar.tgz
# tar -zxf data_original_cedar.tgz  --strip-components=1 -C data/ data/
outermost_dir=$(tar -tzf $FILENAME | head -1 | cut -d'/' -f1)
tar -zxf $FILENAME --strip-components=2 -C data/ "$outermost_dir/data/"
# rm -rf data_original_cedar.tgz
rm -rf $FILENAME