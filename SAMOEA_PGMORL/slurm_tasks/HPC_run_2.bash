#!/bin/bash
#SBATCH --job-name=MORL_TESTS
#SBATCH --account=project_2007911
#SBATCH --time=36:00:00
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=24
#SBATCH --mem=150000

export PATH="/projappl/project_2007911/MORL_tests/conda_env/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/mazumdar/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
srun --ntasks=1 --mem-per-cpu 2000 bash -c "python ../scripts/halfcheetah-v2.py --pgmorl --approach mean" &
srun --ntasks=1 --mem-per-cpu 2000 bash -c "python ../scripts/halfcheetah-v2.py --pgmorl --approach lcb" &
srun --ntasks=1 --mem-per-cpu 2000 bash -c "python ../scripts/halfcheetah-v2.py --pgmorl --approach PMOEA_MOPPO" &
wait
