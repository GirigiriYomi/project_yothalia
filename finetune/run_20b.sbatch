#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=hpml_20b_finetune
#SBATCH --output=%j.out

module purge
module load anaconda3/2020.07
module load openmpi/intel/4.1.1 

source activate /scratch/jd5226/hpml

python prepare.py

torchrun --nproc_per_node 4 finetune.py



