#!/bin/bash
#SBATCH --job-name=MORL_TESTS
#SBATCH --account=project_2007911
#SBATCH --time=36:00:00
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

export PATH="/projappl/project_2007911/MORL_tests/conda_env/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/mazumdar/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
srun python ../scripts/halfcheetah-v2.py --pgmorl
