import itertools
import numpy as np
import pandas as pd
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.ProbRVEA import Probabilistic_RVEA
import torch
import gym
from desdeo_problem.problem.Problem import DataProblem
from desdeo_problem.surrogatemodels.surrogate_RF import RFRegressor
from desdeo_problem.surrogatemodels.surrogate_fullGP import FullGPRegressor
from sample import Sample
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
import pygmo as pg
from desdeo_problem.problem.Variable import variable_builder
from desdeo_problem.problem.Objective import VectorObjective
from desdeo_problem.problem.Problem import MOProblem
from copy import deepcopy

def param_to_nparray(mean_params, params_size):
    x_slice_start = 0
    x_slice_end = 0
    x = np.zeros(params_size)
    for name, param in mean_params.items():
        x_slice_end = x_slice_start + param.nelement()
        x[x_slice_start:x_slice_end] = mean_params[name].clone().cpu().numpy().flatten()
        x_slice_start = x_slice_end           
    return x

def nparray_to_param(mean_params, x, device):
    x_slice_start = 0
    x_slice_end = 0
    for name, param in mean_params.items():
        
        x_slice_end = x_slice_start + param.nelement()
        #if x_slice_end-x_slice_start > 1:
        x_slice = np.reshape(x[x_slice_start:x_slice_end],param.size())
        #else:
        #    x_slice = x[x_slice_start:x_slice_end]
        mean_params[name] = torch.Tensor(x_slice).to(device)
        x_slice_start = x_slice_end    
    return mean_params

def evaluation(policy, ob_rms_mean, ob_rms_var, args):
    eval_env = gym.make(args.env_name)
    eval_env = gym.wrappers.FlattenObservation(eval_env)
    objs = np.zeros(args.obj_num)
    with torch.no_grad():
        for eval_id in range(args.eval_num):
            eval_env.seed(args.seed + eval_id)
            ob = eval_env.reset()
            done = False
            gamma = 1.0
            step = 0
            while not done:
                if args.ob_rms:
                    ob = np.clip((ob - ob_rms_mean) / np.sqrt(ob_rms_var + 1e-8), -10.0, 10.0)
                _, action, _, _ = policy.act(torch.Tensor(ob).unsqueeze(0), None, None, deterministic=True)
                action = action.cpu().detach().numpy().flatten()
                ob, _, done, info = eval_env.step(action)
                objs += gamma * info['obj']
                #print(info['obj'])
                if not args.raw:
                    gamma *= args.gamma
                step += 1
    eval_env.close()
    objs /= args.eval_num
    return objs

def optimize(
        all_offsprings_batch,
        ep,
        device,
        nds_moea_objs,
        nds_moea_policies,
        nds_moea_env_params,
        args):
    
    use_nds_in_model = False
    #merged_individuals = list(itertools.chain.from_iterable(all_offsprings_batch))
    merged_individuals = ep.sample_batch
    sample_policy = merged_individuals[0].actor_critic
    params_state_dict = merged_individuals[0].actor_critic.state_dict()
    mean_params = dict((key, value) for key, value in params_state_dict.items()
                            if ("actor" in key or "fc_mean" in key))
    params_size = 0
    for name, param in mean_params.items():
        params_size += param.nelement()
    pop_size = len(merged_individuals)
    n_objs = args.obj_num
    x = np.zeros((pop_size,params_size))
    y = np.zeros((pop_size,n_objs))
    y_mopg = None
    env_params_mopg = None
    env_params_all = []
    for indiv in range(len(merged_individuals)):
        zz=merged_individuals[indiv].actor_critic.state_dict()
        zz2=dict((key, value) for key, value in zz.items()
                            if ("actor" in key or "fc_mean" in key))
        x[indiv] = param_to_nparray(zz2,params_size)
        y[indiv] = merged_individuals[indiv].objs
        env_params_all.append(merged_individuals[indiv].env_params)
        x_mopg = deepcopy(x)
        y_mopg = deepcopy(y)
        env_params_mopg = deepcopy(env_params_all)
    if nds_moea_objs is not None and not use_nds_in_model:
        x=np.vstack((x,nds_moea_policies))
        y=np.vstack((y,nds_moea_objs))
        for i in nds_moea_env_params:
            env_params_all.append(i)
    # create array of elites
    elite_objs = ep.obj_batch
    elite_pop = np.zeros((len(ep.sample_batch),params_size))
    for indiv in range(len(ep.sample_batch)):
        zz=ep.sample_batch[indiv].actor_critic.state_dict()
        zz2=dict((key, value) for key, value in zz.items()
                            if ("actor" in key or "fc_mean" in key))
        elite_pop[indiv] = param_to_nparray(zz2,params_size)
    if nds_moea_objs is not None:
        elite_objs = np.vstack((elite_objs,nds_moea_objs))
        elite_pop = np.vstack((elite_pop,nds_moea_policies))

    pop_optimized = run_optimizer(n_objs, params_size, x, y, elite_pop, args)
    surrogate_objs = pop_optimized.objectives
    evaluated_solutions = np.zeros((np.shape(pop_optimized.objectives)[0],n_objs))
    env_params_solns = []
    arg_min_dst = np.argmin(euclidean_distances(y_mopg,-surrogate_objs),axis=0)
    for indiv in range(np.shape(pop_optimized.objectives)[0]):
        param = nparray_to_param(mean_params, pop_optimized.individuals[indiv], device)
        sample_policy.load_state_dict(mean_params, strict=False)
        #ob_rms_mean = merged_individuals[arg_min_dst[indiv]].env_params['ob_rms'].mean
        #ob_rms_var = merged_individuals[arg_min_dst[indiv]].env_params['ob_rms'].var
        #env_params_temp = merged_individuals[arg_min_dst[indiv]].env_params
        env_params_temp = env_params_mopg[arg_min_dst[indiv]]
        env_params_solns.append(env_params_temp)
        evaluated_solutions[indiv] = evaluation(sample_policy, 
                                                env_params_temp['ob_rms'].mean, 
                                                env_params_temp['ob_rms'].var, 
                                                args)
    print(evaluated_solutions)
    if nds_moea_objs is None:
        x_combined = pop_optimized.individuals
        y_combined = evaluated_solutions
    else:
        x_combined = np.vstack((pop_optimized.individuals,nds_moea_policies))
        y_combined = np.vstack((evaluated_solutions,nds_moea_objs))
        for i in nds_moea_env_params:
            env_params_solns.append(i)
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = -y_combined)
    nds_objs = y_combined[ndf[0]]
    nds_policies = x_combined[ndf[0]]
    nds_env_params = np.array(env_params_solns)[ndf[0]]
    return nds_objs, nds_policies, nds_env_params

        
def run_optimizer(nobjs, nvars, x, y, p_buf_agents, args):
    y=-y
    #ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = y)
    print("Building surrogates...")
    surrogate_problem = build_surrogates(nobjs, nvars, x, y)
    #print("non dom points:",y[ndf[0]])
    print("Optimizing...")
    population = optimize_surrogates(surrogate_problem, p_buf_agents, args)
    return population

def optimize_surrogates(problem,x, args):
    if args.approach == "mean":
        print("Using mean prediction....")
        evolver_opt =  RVEA(problem=problem,
                    use_surrogates=True,
                    n_iterations=10,
                    lattice_resolution=50,
                    n_gen_per_iter=50 ,
                    reevaluate_parents=False,
                    save_non_dominated=False)
    if args.approach == "lcb":
        print("Using lcb prediction....")
        evolver_opt =  RVEA(problem=problem,
                    selection_type="robust",
                    use_surrogates=True,
                    n_iterations=10,
                    lattice_resolution=50,
                    n_gen_per_iter=50 ,
                    reevaluate_parents=False,
                    save_non_dominated=False)
    elif args.approach == "PMOEA_MOPPO":
        evolver_opt =  Probabilistic_RVEA(problem=problem,
                    use_surrogates=True,
                    n_iterations=10,
                    lattice_resolution=50,
                    n_gen_per_iter=50,
                    reevaluate_parents=False,
                    save_non_dominated=False)
                   
    print("Adding init pop...")
    evolver_opt.population.add(x, use_surrogates=True)
    print("Starting optimization...")
    while evolver_opt.continue_evolution():
        evolver_opt.iterate()
        print("Population size:",np.shape(evolver_opt.population.objectives)[0])
        print("objs:",evolver_opt.population.objectives)
    return evolver_opt.population

def build_surrogates(nobjs, nvars, x_data, y_data):
    x_names = [f'x{i}' for i in range(1,nvars+1)]
    y_names = [f'f{i}' for i in range(1,nobjs+1)]
    row_names = ['lower_bound','upper_bound']
    data = pd.DataFrame(np.hstack((x_data,y_data)), columns=x_names+y_names)
    #x_low = np.ones(nvars)*(np.min(x_data, axis=0)-0.01)
    #x_high = np.ones(nvars)*(np.max(x_data, axis=0)+0.01)
    x_low = np.min(x_data, axis=0)-0.1
    x_high = np.max(x_data, axis=0)+0.1
    bounds = pd.DataFrame(np.vstack((x_low,x_high)), columns=x_names, index=row_names)
    problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names,bounds=bounds)
    problem.train(RFRegressor)
    #problem.train(FullGPRegressor)
    return problem

"""
def optimize_direct(
        all_offsprings_batch,
        ep,
        device,
        nds_moea_objs,
        nds_moea_policies,
        nds_moea_env_params,
        args):
    
    #merged_individuals = list(itertools.chain.from_iterable(all_offsprings_batch))
    merged_individuals = ep.sample_batch
    sample_policy = merged_individuals[0].actor_critic
    params_state_dict = merged_individuals[0].actor_critic.state_dict()
    mean_params = dict((key, value) for key, value in params_state_dict.items()
                            if ("actor" in key or "fc_mean" in key))
    params_size = 0
    for name, param in mean_params.items():
        params_size += param.nelement()
    pop_size = len(merged_individuals)
    n_objs = args.obj_num
    x = np.zeros((pop_size,params_size))
    y = np.zeros((pop_size,n_objs))
    env_params_all = []
    for indiv in range(len(merged_individuals)):
        zz=merged_individuals[indiv].actor_critic.state_dict()
        zz2=dict((key, value) for key, value in zz.items()
                            if ("actor" in key or "fc_mean" in key))
        x[indiv] = param_to_nparray(zz2,params_size)
        y[indiv] = merged_individuals[indiv].objs
        env_params_all.append(merged_individuals[indiv].env_params)
    if nds_moea_objs is not None:
        x=np.vstack((x,nds_moea_policies))
        y=np.vstack((y,nds_moea_objs))
        for i in nds_moea_env_params:
            env_params_all.append(i)
    # create array of elites
    elite_objs = ep.obj_batch
    elite_pop = np.zeros((len(ep.sample_batch),params_size))
    for indiv in range(len(ep.sample_batch)):
        zz=ep.sample_batch[indiv].actor_critic.state_dict()
        zz2=dict((key, value) for key, value in zz.items()
                            if ("actor" in key or "fc_mean" in key))
        elite_pop[indiv] = param_to_nparray(zz2,params_size)
    if nds_moea_objs is not None:
        elite_objs = np.vstack((elite_objs,nds_moea_objs))
        elite_pop = np.vstack((elite_pop,nds_moea_policies))

    pop_optimized = run_optimizer(n_objs, params_size, x, y, elite_pop, args)
    surrogate_objs = pop_optimized.objectives
    evaluated_solutions = np.zeros((np.shape(pop_optimized.objectives)[0],n_objs))
    env_params_solns = []
    arg_min_dst = np.argmin(euclidean_distances(y,-surrogate_objs),axis=0)
    for indiv in range(np.shape(pop_optimized.objectives)[0]):
        param = nparray_to_param(mean_params, pop_optimized.individuals[indiv], device)
        sample_policy.load_state_dict(mean_params, strict=False)
        #ob_rms_mean = merged_individuals[arg_min_dst[indiv]].env_params['ob_rms'].mean
        #ob_rms_var = merged_individuals[arg_min_dst[indiv]].env_params['ob_rms'].var
        #env_params_temp = merged_individuals[arg_min_dst[indiv]].env_params
        env_params_temp = env_params_all[arg_min_dst[indiv]]
        env_params_solns.append(env_params_temp)
        evaluated_solutions[indiv] = evaluation(sample_policy, 
                                                env_params_temp['ob_rms'].mean, 
                                                env_params_temp['ob_rms'].var, 
                                                args)
    print(evaluated_solutions)
    if nds_moea_objs is None:
        x_combined = pop_optimized.individuals
        y_combined = evaluated_solutions
    else:
        x_combined = np.vstack((pop_optimized.individuals,nds_moea_policies))
        y_combined = np.vstack((evaluated_solutions,nds_moea_objs))
        for i in nds_moea_env_params:
            env_params_solns.append(i)
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = -y_combined)
    nds_objs = y_combined[ndf[0]]
    nds_policies = x_combined[ndf[0]]
    nds_env_params = np.array(env_params_solns)[ndf[0]]
    return nds_objs, nds_policies, nds_env_params

def morl_objectives(model, env_name, n_vars, n_objs) -> MOProblem:
    params = dict(
                (key, value)
                for key, value in model.policy.state_dict().items()
                if ("log" in key or "policy" in key or "shared_net" in key or "action" in key)
            )
    
    def vect_f(x):
        return evaluate_vector(env_name=env_name,
                            population=x,
                            model = model,
                            n_objs=n_objs,
                            params=params)

    scale = 100
    x_names = [f'x{i}' for i in range(1,n_vars+1)]
    y_names = [f'f{i}' for i in range(1,n_objs+1)]

    list_vars = variable_builder(x_names,
                                initial_values = np.zeros(n_vars),
                                lower_bounds=np.ones(n_vars)*-scale,
                                upper_bounds=np.ones(n_vars)*scale)

    f_objs = VectorObjective(name=y_names, evaluator=vect_f)
    problem = MOProblem(variables=list_vars, objectives=[f_objs])

    return problem
"""
