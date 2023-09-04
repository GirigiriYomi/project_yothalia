#!/bin/bash
#

module purge
module load anaconda3/2020.07
module load openmpi/intel/4.1.1 

cd llama_core

srun --cpus-per-task=2 --time=3:00:00 --mem=48000 --gres=gpu:2 --pty /bin/bash
source activate llama

