#!/bin/bash
#SBATCH --job-name=benchmark_llama             # Job name
#SBATCH --output=%x.out               # Output file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks (processes) per node
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --mem=64GB                     # Memory per node
#SBATCH --gres=gpu:v100:1
#SBATCH --time=6:00:00               # Max run time (hh:mm:ss)

module purge

# Navigate to the directory where your benchmark code is located
cd /scratch/yc5859/hpml/project_yothalia/yothalia/server

singularity exec --nv \
	    --overlay /scratch/yc5859/hpml/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python benchmark.py"

# you may need to build your own singularity environment
# singularity exec --overlay ../../../my_pytorch.ext3:rw /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash
# srun --cpus-per-task=2 --mem=64GB --gres=gpu:1 --time=2:00:00 --pty /bin/bash