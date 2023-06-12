import os, sys, signal
base_dir = os.path.dirname(os.path.abspath(__file__))
import random
import numpy as np
from multiprocessing import Process, Queue, current_process, freeze_support
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--pgmorl', default=False, action='store_true')
parser.add_argument('--ra', default=False, action='store_true')
parser.add_argument('--pfa', default=False, action='store_true')
parser.add_argument('--moead', default=False, action='store_true')
parser.add_argument('--random', default=False, action='store_true')
parser.add_argument('--num-seeds', type=int, default=5)
parser.add_argument('--num-processes', 
                    type=int, 
                    default=1, 
                    help='number of algorithms to be run in parallel (Note: each algorithm needs 4 * num-tasks processors by default, so the total number of processors is 4 * num-tasks * num-processes.)')
parser.add_argument('--save-dir', type=str, default='/scratch/project_2007911/test_runs/new_mujoco_runs')
parser.add_argument('--approach', type=str, default='mean')
parser.add_argument('--num-processes', 
                    type=int, 
                    default=40, 
                    help='number of algorithms to be run in parallel')
parser.add_argument('--num-runs', type=int, default=3)
parser.add_argument('--runs-folder', type=str, default='/scratch/project_2007911/test_runs/all_runs_hpc_superf')


args = parser.parse_args()

num_runs = args.num_runs
runs_folder = args.runs_folder

random.seed(2000)

approaches = ['mean','lcb','PMOEA_MOPPO']
#envs_index = [0,2,3,4]
envs_index = [1,2,3,4]
env_ids = ['two-way-v0','MO-HalfCheetah-v2','MO-Walker2d-v2','MO-Swimmer-v2','MO-Ant-v2']
total_timesteps = np.array([5e+6, 5e+6, 5e+6, 2e+6, 8e+6], dtype=np.uint32)
nupdates_warmup = np.array([80, 80, 80, 40, 200], dtype=np.uint32)
nupdates_mopg = np.array([20, 20, 20, 10, 40], dtype=np.uint32)

commands = []

for i in envs_index:
    for approach in approaches:    
        for run in range(num_runs):
            seed = random.randint(0, 1000000)
            cmd = 'python {}/../morl/run.py '\
            '--env-name MO-HalfCheetah-v2 '\
            '--seed {} '\
            '--num-env-steps 5000000 '\
            '--warmup-iter 80 '\
            '--update-iter 20 '\
            '--min-weight 0.0 '\
            '--max-weight 1.0 '\
            '--delta-weight 0.2 '\
            '--eval-num 1 '\
            '--pbuffer-num 100 '\
            '--pbuffer-size 2 '\
            '--selection-method prediction-guided '\
            '--num-weight-candidates 7 '\
            '--num-tasks 6 '\
            '--sparsity 1.0 '\
            '--obj-rms '\
            '--ob-rms '\
            '--raw '\
            '--save-dir {}/pgmorl/{}/{}/ '\
            '--approach {}'\
                .format(base_dir, seed, save_dir, approach, i, approach)
        commands.append(cmd)
            cmd = 'python {}/moea_moppo.py '\
                '--env-id {} '\
                '--approach {} '\
                '--runs-folder {} '\
                '--run-num {} '\
                '--seed {} '\
                '--nupdates-warmup {} '\
                '--nupdates-mopg {} '\
                '--total-timesteps {} '\
                '--lattice-res-mopg 5 '\
                '--cuda False '\
                    .format(base_dir,env_ids[i],approach,runs_folder,run+1,seed,nupdates_warmup[i],nupdates_mopg[i],total_timesteps[i])
            commands.append(cmd)
        


def worker(input, output):
    for cmd in iter(input.get, 'STOP'):
        ret_code = os.system(cmd)
        if ret_code != 0:
            output.put('killed')
            break
    output.put('done')

# Create queues
task_queue = Queue()
done_queue = Queue()

# Submit tasks
for cmd in commands:
    task_queue.put(cmd)

# Submit stop signals
for i in range(args.num_processes):
    task_queue.put('STOP')

# Start worker processes
for i in range(args.num_processes):
    Process(target=worker, args=(task_queue, done_queue)).start()

# Get and print results
for i in range(args.num_processes):
    print(f'Process {i}', done_queue.get())
