from desdeo_problem.problem.Variable import variable_builder
from desdeo_problem.problem.Objective import VectorObjective
from desdeo_problem.problem.Problem import MOProblem
from re import L
import gym
import mo_gym
import highway_env
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from stable_baselines3.common.ReferenceVectors import ReferenceVectors
from collections import OrderedDict
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Dict
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes




def param_to_nparray(mean_params, params_size):
    x_slice_start = 0
    x_slice_end = 0
    x = np.zeros(params_size)
    for name, param in mean_params.items():
        x_slice_end = x_slice_start + param.nelement()
        x[x_slice_start:x_slice_end] = mean_params[name].clone().cpu().numpy().flatten()
        x_slice_start = x_slice_end           
    return x


def nparray_to_param(mean_params, x):
    x_slice_start = 0
    x_slice_end = 0
    for name, param in mean_params.items():
        
        x_slice_end = x_slice_start + param.nelement()
        if x_slice_end-x_slice_start > 1:
            x_slice = np.reshape(x[x_slice_start:x_slice_end],param.size())
        else:
            x_slice = x[x_slice_start:x_slice_end]
        #if "bias" in name:
        #    mean_params[name] = th.Tensor(x_slice).cuda()
        #else:
        mean_params[name] = th.Tensor(x_slice).cuda()
        x_slice_start = x_slice_end    
    return mean_params

def evaluate_vector(env_name,
            population,
            model,
            n_objs,
            params,
            is_deter=True):

    num_envs = 1
    pop_size = np.shape(population)[0]
    total_returns = np.zeros((pop_size,n_objs))    
    for indiv in range(pop_size):
        individual = population[indiv,:]
        #indiv_dict = dict((name, individual[param_count]) for name, param_count in range(len(params)))
        indiv_dict = nparray_to_param(params, individual)
        model.policy.load_state_dict(indiv_dict, strict=False)
        env = make_vec_env(env_name, n_envs=num_envs)
        obs = env.reset()
        dones = np.array([False])
        total_returns_indiv = np.zeros((num_envs,n_objs))
        max_num_timesteps = 500
        num_timesteps = 1
        episode_ends = np.full((num_envs), False, dtype=bool)
        #while not np.all(dones):
        
        while num_timesteps <= max_num_timesteps:
            action, _states = model.predict(obs, deterministic=is_deter)
            obs, rewards, dones, info = env.step(action)
            for i in range(num_envs):
                if not episode_ends[i]:
                    total_returns_indiv[i,:] = total_returns_indiv[i,:] + rewards[i]
                if dones[i]:
                    episode_ends[i] = True
            num_timesteps += 1
        total_returns[indiv,:] = np.mean(total_returns_indiv,0)
    return -total_returns

def evaluate_vector_valuefn(env_name,
            population,
            model,
            n_objs,
            params,
            is_deter=True):

    num_envs = 4
    pop_size = np.shape(population)[0]
    pop_rewards = np.zeros((pop_size,n_objs))
    pop_values = np.zeros((pop_size,n_objs))     
    for indiv in range(pop_size):
        individual = population[indiv,:]
        #indiv_dict = dict((name, individual[param_count]) for name, param_count in range(len(params)))
        model = get_models_from_params(model=model, population=individual, params=params)
        #callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1, verbose=1)
        #model.learn(int(1e10), callback=callback_max_episodes)
        model.verbose=0
        model.n_steps=200
        model.learn(total_timesteps=200)
        mean_total_rewards = model.rollout_buffer.rewards.sum(axis=0).mean(axis=0)
        mean_total_values = model.rollout_buffer.values.mean(axis=0)
        pop_rewards[indiv,:] = mean_total_rewards
        pop_values[indiv,:] = mean_total_values        
    return -pop_rewards,-pop_values

def get_models_from_params(model,
                    params,
                    population):
    pop_size = np.shape(population)[0]
    for indiv in range(pop_size):
        individual = population[indiv,:]
        #indiv_dict = dict((name, individual[param_count]) for name, param_count in range(len(params)))
        indiv_dict = nparray_to_param(params, individual)
        indiv_dict["log_std"] = indiv_dict["log_std"]*0
        model.policy.load_state_dict(indiv_dict, strict=False)
    return model
        


def evaluate_scalar(env_name,
            population,
            model,
            n_objs,
            params,
            is_deter=True):
    
    pop_size = np.shape(population)[0]
    total_returns = np.zeros((pop_size,n_objs))    
    for indiv in range(pop_size):
        individual = population[indiv,:]
        #indiv_dict = dict((name, individual[param_count]) for name, param_count in range(len(params)))
        indiv_dict = nparray_to_param(params, individual)
        model.policy.load_state_dict(indiv_dict, strict=False)
        env = mo_gym.make(env_name)
        obs = env.reset()
        done = False        
        while not done:
            action, _states = model.predict(obs, deterministic=is_deter)
            #print(action)
            obs, rewards, done, info = env.step(action)
            total_returns[indiv,:] = total_returns[indiv,:] + rewards
            #print(total_returns)
    return -total_returns

def morl_objectives(model, env_name, n_vars, n_objs) -> MOProblem:
    """
    PPO model params, n_objs as init
    input nparray population
    output nparray objective values
    """
   
    """
    def vect_f(x):
        if isinstance(x, list):
            if len(x) == n_vars:
                return [objective_unc(x)]
            elif len(x[0]) == n_vars:
                return list(map(objective_unc, x))
        else:
            if x.ndim == 1:
                return [objective_unc(x)]
            elif x.ndim == 2:
                return list(map(objective_unc, x))
        raise TypeError("Unforseen problem, contact developer")
    """

    params = dict(
                (key, value)
                for key, value in model.policy.state_dict().items()
                if ("log" in key or "policy" in key or "shared_net" in key or "action" in key)
            )
    params["log_std"] = params["log_std"]*0
    #"log" in key or
    #env = model.get_env()

    def vect_f(x):
        """
        return evaluate_scalar(env_name=env_name,
                            population=x,
                            model = model,
                            n_objs=n_objs,
                            params=params)
        """
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



    