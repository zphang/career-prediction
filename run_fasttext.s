#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:10:00
#SBATCH --mem=50GB
#SBATCH --job-name=fasttext
#SBATCH --mail-type=END
#SBATCH --mail-user=mf3490@nyu.edu
#SBATCH --output=fasttext_%j.out
 
#module purge
module load pytorch/python3.6/0.2.0_3
source activate jobembeddings
RUNDIR=$SCRATCH/data/jobembeddings
$RUNDIR/fastText/fasttext supervised -input $RUNDIR/json_sample10m_fasttext_train -output $RUNDIR/fasttext_10m_model
