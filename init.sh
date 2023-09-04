#!/bin/bash
#

module purge
module load anaconda3/2020.07
module load openmpi/intel/4.1.1 

srun --cpus-per-task=1 --time=3:00:00 --mem=32000 --gres=gpu:1 --pty /bin/bash
