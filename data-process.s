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

python $RUNDIR/career-prediction/data-process.py
