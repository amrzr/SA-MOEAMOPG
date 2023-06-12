'''
Visualize the training process.
'''
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import os
from copy import deepcopy
from multiprocessing import Process
import sys
import pygmo as pg
import pandas as pd
from matplotlib import rc
import seaborn as sns; sns.set()
# np.set_printoptions(precision=1)
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
#rc('font',**{'family':'serif','serif':['Helvetica']})
#rc('text', usetex=True)

def get_ep_indices(obj_batch):
    # return sorted indices of undominated objs
    if len(obj_batch) == 0: return np.array([])
    sort_indices = np.lexsort((obj_batch.T[1], obj_batch.T[0]))
    ep_indices = []
    max_val = -np.inf
    for idx in sort_indices[::-1]:
        if obj_batch[idx][1] > max_val:
            max_val = obj_batch[idx][1]
            ep_indices.append(idx)
    return ep_indices[::-1]

# compute the hypervolume and sparsity given the pareto points, only for 2-dim objectives now
def compute_metrics(obj_batch):
    if obj_batch.shape[1] != 2:
        return 0
    
    objs = obj_batch
    if np.shape(objs)[0]>=2:
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = -objs)
        objs = objs[ndf[0]]
    objs = objs[objs[:, 0].argsort()]
    ref_x, ref_y = 0.0, 0.0 # set referent point as (0, 0)
    x, hypervolume = ref_x, 0.0
    sqdist = 0.0
    for i in range(len(objs)):
        hypervolume += (max(ref_x, objs[i][0]) - x) * (max(ref_y, objs[i][1]) - ref_y)
        x = max(ref_x, objs[i][0])
        if i > 0:
            sqdist += np.sum(np.square(objs[i] - objs[i - 1]))

    if len(objs) == 1:
        sparsity = 0.0
    else:
        sparsity = sqdist / (len(objs) - 1)

    #print('Pareto size : {}, Hypervolume : {:.0f}, Sparsity : {:.2f}'.format(len(objs), hypervolume, sparsity))

    return hypervolume, sparsity

def get_objs(objs_path):
    objs = []
    if os.path.exists(objs_path):
        with open(objs_path, 'r') as fp:
            data = fp.readlines()
            for j, line_data in enumerate(data):
                line_data = line_data.split(',')
                objs.append([float(line_data[0]), float(line_data[1])])
    return objs

approaches = ['mean','lcb','PMOEA_MOPPO']
approaches_disp = ['Mean','LCB','PMOEA']
envs = ['MO-HalfCheetah-v2','MO-Swimmer-v2','MO-Walker2d-v2']
len_runs = 2

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='MO-HalfCheetah-v2')
parser.add_argument('--log-dir', type=str, default='/scratch/project_2007911/test_runs/new_mujoco_runs_1/')
parser.add_argument('--save-fig', default=False, action='store_true')
parser.add_argument('--title', type=str, default=None)
parser.add_argument('--obj', type=str, nargs='+', default=None)
args = parser.parse_args()

log_dir_root = args.log_dir

fig = plt.figure()
fig.set_size_inches(20, 5*len(envs))
#fig2 = plt.figure()
#fig2.set_size_inches(2*len(envs), 2)
for e in range(len(envs)):
    points_sequence_df = []
    pareto_df = []
    args.env = envs[e]
    if args.env in ['MO-HalfCheetah-v2', 'MO-Walker2d-v2', 'MO-Swimmer-v2', 'MO-Humanoid-v2']:
        args.obj = ['Forward_Speed', 'Energy_Efficiency']
    elif args.env == 'MO-Ant-v2':
        args.obj = ['X-Axis_Speed', 'Y-Axis_Speed']
    elif args.env == 'MO-Hopper-v2':
        args.obj = ['Running_Speed', 'Jumping_Height']
    for run in range(len_runs):
        for z in range(len(approaches)):
            iterations_str = []
            iterations = []
            ep_objs = []
            population_objs = []
            elites_objs = []
            elites_weights = []
            predictions = []
            offsprings = []
            elite_objs_moea = []
            args.log_dir = log_dir_root + envs[e] + '/pgmorl/' + approaches[z] + '/' + str(run)
            all_iteration_folders = os.listdir(args.log_dir)
            for folder in all_iteration_folders:
                if os.path.isdir(os.path.join(args.log_dir, folder)) and folder != 'final':
                    iterations.append(int(folder))
                    population_log_dir = os.path.join(args.log_dir, folder, 'population')
                    # load population objs
                    population_objs.append(get_objs(os.path.join(population_log_dir, 'objs.txt')))
                    # load ep
                    ep_log_dir = os.path.join(args.log_dir, folder, 'ep')
                    ep_objs.append(get_objs(os.path.join(ep_log_dir, 'objs.txt')))
                    # load elites
                    elites_log_dir = os.path.join(args.log_dir, folder, 'elites')
                    elites_objs.append(get_objs(os.path.join(elites_log_dir, 'elites.txt')))
                    elites_moea_dir = os.path.join(args.log_dir, folder, 'moea')
                    elite_objs_moea.append(get_objs(os.path.join(elites_moea_dir, 'objs.txt')))
                    elites_weights.append(get_objs(os.path.join(elites_log_dir, 'weights.txt')))
                    predictions.append(get_objs(os.path.join(elites_log_dir, 'predictions.txt')))
                    offsprings.append(get_objs(os.path.join(elites_log_dir, 'offsprings.txt')))

            for weights in elites_weights:
                for weight in weights:
                    norm = np.sqrt(weight[0] ** 2 + weight[1] ** 2)
                    weight[0] /= norm
                    weight[1] /= norm
                    
            iterations = np.array(iterations)
            ep_objs = np.array(ep_objs)
            population_objs = np.array(population_objs)
            elites_objs = np.array(elites_objs)
            elite_objs_moea = np.array(elite_objs_moea)
            elites_weights = np.array(elites_weights)
            predictions = np.array(predictions)
            offsprings = np.array(offsprings)

            have_pred = (predictions.size > 0)
            have_offspring = (offsprings.size > 0)

            sorted_index = np.argsort(iterations)
            sorted_ep_objs = []
            sorted_population_objs = []
            sorted_elites_objs = []
            sorted_elites_weights = []
            sorted_predictions = []
            sorted_offsprings = []
            sorted_elite_moea_objs = []
            sorted_all_objs = []
            utopians = []
            for i in range(len(sorted_index)):
                index = sorted_index[i]
                sorted_ep_objs.append(deepcopy(ep_objs[index]))
                sorted_population_objs.append(deepcopy(population_objs[index]))
                sorted_elites_objs.append(deepcopy(elites_objs[index]))
                sorted_elite_moea_objs.append(deepcopy(elite_objs_moea[index]))
                #sorted_all_objs.append(deepcopy(np.vstack((ep_objs[index],elite_objs_moea[index]))))
                sorted_all_objs.append(deepcopy(elite_objs_moea[index]))
                sorted_elites_weights.append(deepcopy(elites_weights[index]))
                if have_pred:
                    sorted_predictions.append(deepcopy(predictions[index]))
                if have_offspring:
                    if i < len(sorted_index) - 1:
                        sorted_offsprings.append(deepcopy(offsprings[sorted_index[i + 1]]))
                    else:
                        sorted_offsprings.append([])
                utopian = np.max(sorted_ep_objs[i], axis=0)
                utopians.append(utopian)
            
            if run == 0:
                for i in range(len(elite_objs_moea[index])):
                    pareto_df.append([approaches_disp[z], elite_objs_moea[index][i][0],elite_objs_moea[index][i][1]])

            all_elites_objs = []
            all_elites_weights = []
            for i in range(len(sorted_elites_objs)):
                for j in range(len(sorted_elites_objs[i])):
                    all_elites_objs.append(sorted_elites_objs[i][j])
                    all_elites_weights.append(sorted_elites_weights[i][j])
            all_elites_objs = np.array(all_elites_objs)
            all_elites_weights = np.array(all_elites_weights)

            hypervolumes, sparsities = [], []
            hypervolumes_all, sparsities_all = [], []
            for i in range(len(sorted_ep_objs)):
                hypervolume, sparsity = compute_metrics(np.array(sorted_ep_objs[i]))
                hypervolumes.append(hypervolume)
                sparsities.append(sparsity)
                hypervolume_all, sparsity_all = compute_metrics(np.array(sorted_all_objs[i]))
                hypervolumes_all.append(hypervolume_all)
                sparsities_all.append(sparsity_all)

            for i in range(len(hypervolumes)):
                #if z == 0:
                #    points_sequence_df.append([i, run, 'MOPG', hypervolumes[i],sparsities[i]])
                points_sequence_df.append([i, run, approaches_disp[z], hypervolumes_all[i],sparsities_all[i]])
                
                #points_sequence_df.append([i, run, approaches_disp[z], hypervolumes_all[i] - hypervolumes[i],sparsities_all[i] - sparsities[i]])
                
                #points_sequence_df.append([i, run, approaches_disp[z], hypervolumes_all[i],sparsities_all[i]])
            
            #print(epoch_hypervolume_drawing)
            #print(np.array(hypervolumes_all) - np.array(hypervolumes))
    #print(points_sequence_df)
    points_sequence_dfpd = pd.DataFrame(points_sequence_df, columns=['Iterations', 'Run', 'Approaches', 'Hypervolume', 'Sparsity'])
    pareto_dfpd = pd.DataFrame(pareto_df, columns=['Approaches', args.obj[0],args.obj[1]])        
    color_map = plt.cm.get_cmap('viridis')
    color_map = color_map(np.linspace(0, 1, len(approaches)))
    
    #fig.set_size_inches(5, 4)
    ax = fig.add_subplot(len(envs), 3, 3*e+1)
    #ax.legend(loc='lower right')
    #ax.get_legend().remove()
    ax = sns.lineplot(x="Iterations", y="Hypervolume",
                    hue="Approaches", style="Approaches", #err_style="bars",
                    markers=True, dashes=False, data=points_sequence_dfpd, palette=color_map).set(title=envs[e])
    ax = fig.add_subplot(len(envs), 3, 3*e+2)
    #ax.legend(loc='upper right')
    ax = sns.lineplot(x="Iterations", y="Sparsity",
                hue="Approaches", style="Approaches", #err_style="bars",
                markers=True, dashes=False, data=points_sequence_dfpd, palette=color_map).set(title=envs[e])
    #ax.set(yscale="log")
    ax = fig.add_subplot(len(envs), 3, 3*e+3)
    ax = sns.scatterplot(data=pareto_dfpd, x=args.obj[0], y=args.obj[1], hue="Approaches",palette=color_map).set(title=envs[e])

fig.savefig(log_dir_root+'comparison_plots.pdf', bbox_inches='tight')
ax.clear() 
    

