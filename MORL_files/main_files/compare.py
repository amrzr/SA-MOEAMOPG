import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
import numpy as np
import pickle
import argparse

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--approach", type=str, default="PMOEA_MOPPO",
        help="the name of this experiment")
    parser.add_argument("--runs-folder", type=str, default="all_runs",
        help="the name of this experiment")
    parser.add_argument("--run-num", type=int, default=3,
        help="run number for multiple runs")
    parser.add_argument("--env-id", type=str, default="MO-HalfCheetah-v2",
        help="the id of the environment")
    args = parser.parse_args()
    return args

args = parse_args()
folder = base_dir+ '/' + args.runs_folder + '/' + args.approach + '/' + args.env_id + '/'+ str(args.run_num)
with open(folder+'/archive_data_iteration.pickle', 'rb') as handle:
        archive_data_iteration=pickle.load(handle)
print(archive_data_iteration)