#!/bin/bash
#SBATCH --job-name=MORL_TESTS
#SBATCH --account=project_2007911
#SBATCH --time=36:00:00
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=24

export PATH="/projappl/project_2007911/MORL_tests/conda_env/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/mazumdar/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


srun --ntasks=1 --mem-per-cpu 1500 bash -c "python ../scripts/walker2d-v2.py --pgmorl --approach lcb --save-dir /scratch/project_2007911/test_runs/new_mujoco_xx/MO-Walker2d-v2" &
srun --ntasks=1 --mem-per-cpu 1500 bash -c "python ../scripts/halfcheetah-v2.py --pgmorl --approach lcb --save-dir /scratch/project_2007911/test_runs/new_mujoco_xx/MO-HalfCheetah-v2" &
srun --ntasks=1 --mem-per-cpu 1500 bash -c "python ../scripts/highway-v0.py --pgmorl --approach lcb --save-dir /scratch/project_2007911/test_runs/new_mujoco_xx/two-way-v0" &
srun --ntasks=1 --mem-per-cpu 1500 bash -c "python ../scripts/ant-v2.py --pgmorl --approach lcb --save-dir /scratch/project_2007911/test_runs/new_mujoco_xx/MO-Ant-v2" &
srun --ntasks=1 --mem-per-cpu 1500 bash -c "python ../scripts/swimmer-v2.py --pgmorl --approach lcb --save-dir /scratch/project_2007911/test_runs/new_mujoco_xx/MO-Swimmer-v2" &
wait
