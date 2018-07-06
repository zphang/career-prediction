#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=250GB
#SBATCH --output=fasttext_%j.out
#SBATCH --time=10:10:00
module load pytorch/python3.6/0.2.0_3
source activate nlpclass
RUNDIR=/home/zp489/scratch/data/jobembeddings

python $RUNDIR/career-prediction/extract_fields.py \
    --output $RUNDIR/json_sample10m_fasttext_train \
    --corpus $RUNDIR/json_usa_nyu_09142017__10mil_filtered.txt \
    --grams_file $RUNDIR/data_title_words_unigram.pickle \
    --stopwords_file $RUNDIR/career-prediction/stopwords.tsv
