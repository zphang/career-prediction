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
source $SCRATCH/nlp-project-data/py3.6.3/bin/activate
RUNDIR=$SCRATCH/independent-study-data
$RUNDIR/data/fastText-0.1.0/fasttext supervised -input $RUNDIR/data/sample_data/json_sample10m_fasttext_train -output $RUNDIR/data/sample_data/fasttext_10m_model
#$RUNDIR/data/fastText-0.1.0/fasttext test $RUNDIR/data/sample_data/fasttext_1m_model.bin $RUNDIR/data/sample_data/json_sample1m_fasttext_test 5
