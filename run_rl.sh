#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=bash
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=v100l:2
#SBATCH --mem=128G

#SBATCH --time=11:0:0
#SBATCH --account=def-gpenn

if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    echo "Example: $0 myfile.py"
    exit 1
fi

./compute_can_setup.sh $1
cd $SLURM_TMPDIR
module purge 
module load opencv cuda gcc scipy-stack 
source env/bin/activate

#move code
cp /home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/*.py .

#python models.py --discrete --model_path /home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/lambdabertmodel_linear_last/best_linear_last.ckpt --save_dir /home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/discrete_linear_last/ --batch_size 20 --finetune_discrete #--model_is_discrete #--bert_is_last
# export BERT_TYPE="roberta_base"
# python models.py --save_dir "/home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/roberta_base_model/" --batch_size 150 --bert_is_last --custom_transformer

echo "----File is $1----"

export BERT_TYPE="bert_base"
# python models.py --model_path "/home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/bert_base_model/best_bert_base_r2.ckpt" --save_dir "/home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/bert_base_model/" --batch_size 150 --bert_is_last --custom_transformer
python models.py --save_dir "~/scratch/bb11_filt_model/" --batch_size 150 --bert_is_last --custom_transformer
# python models.py --save_dir "/home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/bert_base_filtered_model" --batch_size 150 --bert_is_last --custom_transformer

# export BERT_TYPE="multilingual_bert"
# python models.py --save_dir "/home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/simple_lambda_model/" --batch_size 150 --bert_is_last --custom_transformer
# python models.py --model_path "/home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/multilingual_base/best_multilingual_base.ckpt" --save_dir "/home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/multilingual_base/" --batch_size 150 --bert_is_last --custom_transformer
# python models.py --save_dir "/home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/linear_last/" --model_path /home/mishaalk/projects/def-gpenn/mishaalk/lambdaBERT/linear_last/train_r1_lin_last.ckpt --batch_size 20 --bert_is_last #--custom_transformer
# python models.py --shuffled_mode --t_force --save_dir "/home/mishaalk/scratch/lambdaModelsNoTforce/" --batch_size 30
# python models.py --shuffled_mode --save_dir "/home/mishaalk/scratch/lambdaPosModel/" --model_path /home/mishaalk/scratch/bestpost.ckpt
# rem,ember not T Force is ur model w discrete

# PREVIOUSLT  --cpus-per-task=1

#salloc --time=1:30:0 --ntasks=12 --gres=gpu:v100l:2 --mem=128G --nodes=1 --account=def-gpenn
