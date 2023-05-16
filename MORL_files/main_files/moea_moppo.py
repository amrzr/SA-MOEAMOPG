################################################
#########  MORL + MOEA with critic and actors in the SAME class
#########  Build Random Forest surrogate model of actors and episodic returns
#########  Run MOEA on surrogate and inject actors to MOPG
#########  Two performance buffer for storing MOEA and MOPG solutions with APD scalarization
#########  RF model built with the performance buffer data
#########  Modified critic loss as norm(V_pi-V)^2
#########  Advantage is scalarized (ASF). Returns and values are not scalarized.
################################################
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
import warnings
warnings.filterwarnings("ignore")
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'env_mujoco/'))
import argparse
import random
import time
from distutils.util import strtobool

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_problem.problem.Problem import DataProblem
from desdeo_problem.surrogatemodels.surrogate_RF import RFRegressor
from desdeo_problem.surrogatemodels.surrogate_fullGP import FullGPRegressor
import pandas as pd
import pygmo as pg

import mo_gym
import env_mujoco.environments
import gym
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import mo_utils
from ReferenceVectors import ReferenceVectors
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.RVEA import oRVEA
from desdeo_emo.EAs.ProbRVEA import Probabilistic_RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_emo.population.Population import Population
from desdeo_problem.problem.Variable import variable_builder
from desdeo_problem.problem.Objective import VectorObjective
from desdeo_problem.problem.Problem import MOProblem
import pickle
import highway_env
import mopg_worker
import matplotlib.pyplot as plt
import APD_performance

#is_highway = False
#is_highway = True
#id='two-way-v0'
#id = "MO-HalfCheetah-v2"

config = {
    "observation": {
            "vehicles_count": 4, # including ego
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "type": "Kinematics",
            "normalize": False,
            "absolute": False,
            "clip":False
        },
     "action": {
        "type": "ContinuousAction"
    },
    'offroad_terminal': True,
    "simulation_frequency": 60,
    "policy_frequency": 60,
    "collision_reward": -10,
    "on_road_reward": -10
}

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--approach", type=str, default="PMOEA_MOPPO",
        help="the name of this experiment")
    parser.add_argument("--runs-folder", type=str, default="all_runs",
        help="the name of this experiment")
    parser.add_argument("--run-num", type=int, default=1,
        help="run number for multiple runs")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="MORL_Tests",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="ai4la",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--load-data", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="load the warmup data")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MO-HalfCheetah-v2",
        help="the id of the environment")
    parser.add_argument("--lattice-res-mopg", type=int, default=5,
        help="lattice resolution for mopg")
    parser.add_argument("--lattice-res-pbuf", type=int, default=100,
        help="lattice resolution for performance buffer")
    parser.add_argument("--num-objs", type=int, default=2,
        help="number of objectives")
    parser.add_argument("--nupdates-warmup", type=int, default=80,
        help="number of updates in warmup stage")
    parser.add_argument("--nupdates-mopg", type=int, default=20,
        help="number of updates in later optimization stage")
    parser.add_argument("--max-iters", type=int, default=25,
        help="maximum number of iteration")
    parser.add_argument("--total-timesteps", type=int, default=5e+6,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.995,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        #if config is not None:
        if args.env_id=='two-way-v0':
            env.configure(config)
        env.reset(seed=args.seed)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #env = gym.wrappers.RecordVideo(env, video_folder=folder+"/videos", episode_trigger=lambda e: True)
        env = gym.wrappers.ClipAction(env)        
        #env = gym.wrappers.NormalizeObservation(env)
        #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        #env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

def get_pbi_weights(solution, ideal_point, weights, theta):
    
    print("given weights:",weights)
    F = -solution.clone().cpu().numpy()
    w = np.array(weights)
    #w2 = -w+1
    #w2=w2/np.sum(w2)
    w2=w
    z_star = -np.array(ideal_point)
    d1_vect = (F - z_star)* w2 / np.linalg.norm(w2)
    d1_vect_mag = np.dot((F - z_star), w2)
    d1 = np.linalg.norm(d1_vect_mag) / np.linalg.norm(w2)
    d2_vect = (F-z_star) - ((d1 * w2)/np.linalg.norm(w2))
    d2 = np.linalg.norm(d2_vect)
    theta = 0
    pbi_direction_vect = d1_vect + theta * d2_vect
    print("d1:",d1_vect)
    #print("d2:",d2_vect)
    #print("theta:",theta)
    ws_direction_vect = F*w
    pbi_direction_norm = pbi_direction_vect/np.linalg.norm(pbi_direction_vect)
    pbi_modified_weights=pbi_direction_norm/np.sum(pbi_direction_norm)
    #pbi_modified_weights = -pbi_modified_weights+1
    #pbi_modified_weights = pbi_modified_weights/np.sum(pbi_modified_weights)
    return pbi_modified_weights

def scalarize_ASF_tensor(fitness_vector, 
                weights, 
                nadir_point,
                utopian_point,
                num_envs,
                rho = 1e-6):

    f = -fitness_vector.clone().cpu().numpy()
    rho = rho
    z_nad = -nadir_point.clone().cpu().numpy()
    z_uto = -utopian_point.clone().cpu().numpy()
    mu = weights.reshape(1,-1,1)
    z_uto = z_uto.reshape(1,-1,1)
    z_nad = z_nad.reshape(1,-1,1)
    scale = z_nad - z_uto
    max_term = np.max(mu * (f - z_uto)/scale, axis=1)
    sum_term = rho * np.sum((f - z_uto)/scale, axis=1)
    #max_term = np.max(mu * f, axis=1)
    #sum_term = rho * np.sum(f, axis=1)
    return torch.Tensor(-(max_term + sum_term)).to(device)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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

def collect_rollouts(env_eval, agent, device, args):
    next_obs = env_eval.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)
    num_objs = args.num_objs
    rewards = torch.zeros((args.num_steps, num_objs, 1)).to(device)
    dones = torch.zeros((args.num_steps, 1)).to(device)
    done = False
    step = 0
    while not done:
        dones[step] = next_done
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(next_obs, 0, action="deterministic") 
        next_obs, _, terminated, infos = env_eval.step(action.cpu().numpy())
        reward = infos['reward']
        reward = np.vstack(reward[:]).astype(np.float32)
        done = terminated
        rewards[step] = torch.tensor(reward.transpose()).to(device)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
        step += 1
    episodic_total_return = rewards.sum(2).sum(0)#/(torch.sum(dones) + (1-next_done))
    return episodic_total_return.cpu().detach().numpy()

def evaluate_vector(env_eval, pop, model, n_objs, params, device):
    avg_return = np.zeros((np.shape(pop)[0],n_objs))
    for i in range(np.shape(pop)[0]):
        indiv_dict = nparray_to_param(params, pop[i,:], device)
        model.actor_mean.load_state_dict(indiv_dict, strict=False)
        avg_return[i,:] = collect_rollouts(env_eval, model, device, args)
    return avg_return

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
                    layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)), #+args.num_objs
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 64)),
                    nn.Tanh(),
                    #layer_init(nn.Linear(64, 64)),
                    #nn.Tanh(),
                    layer_init(nn.Linear(64, args.num_objs), std=1.0),
                    )
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        
        self.expected_episodic_return = np.zeros(args.num_objs)

    def get_value(self, x, weight):
        return self.critic(x) #torch.cat((x,weight),1)

    def get_action_and_value(self, x, weight, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        elif action == 'deterministic':
            action = action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def run_optimizer(nobjs, nvars, x, y,p_buf_agents):
    y=-y
    #ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = y)
    print("Building surrogates...")
    surrogate_problem = build_surrogates(nobjs, nvars, x, y)
    #print("non dom points:",y[ndf[0]])
    print("Optimizing...")
    population = optimize_surrogates(surrogate_problem,p_buf_agents)
    return population

def optimize_surrogates(problem,x):
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
                    n_gen_per_iter=50 ,
                    reevaluate_parents=False,
                    save_non_dominated=False)
                   
    print("Adding init pop...")
    evolver_opt.population.add(x, use_surrogates=True)
    print("Starting optimization...")
    while evolver_opt.continue_evolution():
        evolver_opt.iterate()
        print("Population size:",np.shape(evolver_opt.population.objectives)[0])
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

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # monitor_gym=True, no longer works for gymnasium
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    #device = torch.device("cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    env_eval = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(1)]
    )
    assert isinstance(env_eval.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent_base = Agent(envs).to(device)    
    mean_params = dict(
        (key, value)
        for key, value in agent_base.actor_mean.state_dict().items()
    )

    params_size = 0
    for name, param in mean_params.items():
        params_size += param.nelement()

    mean_params_critic = dict(
        (key, value)
        for key, value in agent_base.critic.state_dict().items()
    )

    params_size_critic = 0
    for name, param in mean_params_critic.items():
        params_size_critic += param.nelement()
    
    
    #### relevant variables
    num_vars = params_size
    #num_updates = args.total_timesteps // args.batch_size
    num_updates = args.nupdates_mopg
    max_iters = args.max_iters    
    num_objs = args.num_objs
    weight_vect_obj = ReferenceVectors(number_of_objectives=args.num_objs, lattice_resolution=args.lattice_res_mopg)
    weight_vectors_spherical = weight_vect_obj.values
    weight_vectors = weight_vect_obj.values_planar
    #weight_vectors = np.array([[1,0],[0.5,0.5],[0,1]])
    num_weight_vectors = np.shape(weight_vectors)[0]
    weight_vect_perf = ReferenceVectors(number_of_objectives=args.num_objs, lattice_resolution=args.lattice_res_pbuf)
    ref_vectors_performance = weight_vect_perf.values
    refx = weight_vect_perf.neighbouring_angles_current
    mopg_step = np.zeros(num_weight_vectors)
    global_step = 0


    folder = args.runs_folder + '/' + args.approach + '/' + args.env_id + '/'+ str(args.run_num)
    try:
        os.makedirs(folder, exist_ok = True)
    except OSError:
        pass
    
    agents = []    
    for i in range(num_weight_vectors):
        ax=Agent(envs).to(device)        
        agents.append(ax)
    
    #agents_archive = []
    #episodic_return_archive = []
    #critics_archive = []

    a_arch_np = []
    ret_arch_np = []
    critics_archive_np = []
    weights_archive_np = []
    
    p_buf_agents = []
    p_buf_critics = []
    p_buf_ret = []
    p_buf_weights = []

    
    p_buf_moea_agents = []
    p_buf_moea_ret = []

    nds_agents = []
    nds_ret = []
    nds_moea_agents = []
    nds_moea_ret = []

    x_moea = []
    y_moea = []

    archive_data_iteration = {}
    #ep_return_final = collect_rollouts(envs, agent_base, device, args)
    mean_params = dict((key, value) for key, value in agent_base.actor_mean.state_dict().items())
    mean_params_critic = dict((key, value) for key, value in agent_base.critic.state_dict().items())
    agent_vars = param_to_nparray(mean_params,params_size)
    critic_vars = param_to_nparray(mean_params_critic, params_size_critic)
    #agents_archive.append(agent_vars)
    #episodic_return_archive.append(ep_return_final)
    #critics_archive.append(critic_vars)

    archive_data = None
    #for i in range(num_weight_vectors):
    #    with open(folder+'/moppo_actors_data'+str(n)+'_'+str(i)+'.pickle', 'rb') as handle:
    #        a[i].load_state_dict(pickle.load(handle))
    if args.load_data:
        with open(folder+'/archive_data_warmup.pickle', 'rb') as handle:
             archive_data=pickle.load(handle)
        a_arch_np= archive_data["agents_archive"]
        ret_arch_np=archive_data["episodic_return_archive"]
        critics_archive_np=archive_data["critics_archive_np"]
        weights_archive_np=archive_data["weights_archive_np"]
        p_buf_agents=archive_data["p_buf_agents"]
        p_buf_critics=archive_data["p_buf_critics"]
        p_buf_ret=archive_data["p_buf_ret"]
        p_buf_weights=archive_data["p_buf_weights"]
        nds_agents=archive_data["nds_agents"]
        nds_ret=archive_data["nds_ret"]

    #MOEA initilize
    mopg_step = np.zeros(num_weight_vectors)
    critic_universal = None
    max_updates = args.total_timesteps // args.num_steps // args.num_envs
    #max_updates = args.nupdates_warmup + num_updates * (max_iters-1)
    tot_steps = 0
    n_iters = 0

    while tot_steps<=args.total_timesteps:
        print("********** Iteration **********",n_iters)    
        if n_iters==0:
            num_updates_mopg = args.nupdates_warmup
        else:
            num_updates_mopg = num_updates
        
        for w in range(num_weight_vectors):
            if args.load_data and n_iters==0:
                with open(folder+'/moppo_agents_warmup_'+str(w)+'.pickle', 'rb') as handle:
                    agents[w].load_state_dict(pickle.load(handle))
            else:
                agent_mopg, agents_archive_mopg, critics_archive_mopg, ep_return_archive = mopg_worker.run_mopg(agents[w],
                                                                    weight_vectors[w],
                                                                    num_updates_mopg, 
                                                                    envs, 
                                                                    env_eval, 
                                                                    global_step,
                                                                    device,
                                                                    tot_steps,
                                                                    max_updates, 
                                                                    args)
                agents[w] = agent_mopg
                
                #mean_params = dict((key, value) for key, value in agent_mopg.actor_mean.state_dict().items())
                #mean_params_critic = dict((key, value) for key, value in agent_mopg.critic.state_dict().items())
                #agent_vars = param_to_nparray(mean_params,params_size)
                #agent_vars = np.append(agent_vars, weight_vectors[w])
                #critic_vars = param_to_nparray(mean_params_critic, params_size_critic)
                #agents_archive.append(agents_archive_mopg)
                #episodic_return_archive.append(ep_return_archive)
                #critics_archive.append(critic_vars)
                if n_iters == 0 and w ==0:
                    a_arch_np = agents_archive_mopg
                    ret_arch_np = ep_return_archive
                    critics_archive_np = critics_archive_mopg
                    weights_archive_np = np.tile(weight_vectors[w],[num_updates_mopg,1])
                    p_buf_agents = agents_archive_mopg
                    p_buf_critics = critics_archive_mopg
                    p_buf_ret = ep_return_archive
                    p_buf_weights = weights_archive_np
                    nds_agents = agents_archive_mopg
                    nds_ret = ep_return_archive
                else:
                    a_arch_np = np.vstack((a_arch_np,agents_archive_mopg))
                    ret_arch_np = np.vstack((ret_arch_np,ep_return_archive))
                    critics_archive_np = np.vstack((critics_archive_np, critics_archive_mopg))
                    weights_archive_np = np.vstack((weights_archive_np,np.tile(weight_vectors[w],[num_updates_mopg,1])))
                    p_buf_agents = np.vstack((p_buf_agents,agents_archive_mopg))
                    p_buf_critics = np.vstack((p_buf_critics,critics_archive_mopg))
                    p_buf_ret = np.vstack((p_buf_ret,ep_return_archive))
                    p_buf_weights = np.vstack((p_buf_weights,np.tile(weight_vectors[w],[num_updates_mopg,1])))
                    nds_agents = np.vstack((nds_agents,agents_archive_mopg))
                    nds_ret = np.vstack((nds_ret,ep_return_archive))


        tot_steps += num_updates_mopg*args.num_steps*args.num_envs
        global_step = tot_steps * num_weight_vectors
        print()

        if args.load_data and n_iters == 0:
            pass
        else:    
            performance_buffer = APD_performance.APD_performance(p_buf_ret,
                                                                ref_vectors_performance,
                                                                #(n_iters+1)/max_iters,
                                                                0.001,
                                                                num_objs,
                                                                refx
                                                                )
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = -nds_ret)

            p_buf_agents = p_buf_agents[performance_buffer]
            p_buf_ret = p_buf_ret[performance_buffer]
            p_buf_weights = p_buf_weights[performance_buffer]
            p_buf_critics = p_buf_critics[performance_buffer]

            nds_agents = nds_agents[ndf[0]]
            nds_ret = nds_ret[ndf[0]]

            archive_data = {"agents_archive":a_arch_np,
                            "episodic_return_archive":ret_arch_np,
                            "critics_archive_np":critics_archive_np,
                            "weights_archive_np":weights_archive_np,
                            "p_buf_agents":p_buf_agents,
                            "p_buf_critics":p_buf_critics,
                            "p_buf_ret":p_buf_ret,
                            "p_buf_weights":p_buf_weights,
                            "nds_agents":nds_agents,
                            "nds_ret":nds_ret}

        if not args.load_data and n_iters == 0:
            with open(folder+'/archive_data_warmup.pickle', 'wb') as handle:
                pickle.dump(archive_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            for i in range(num_weight_vectors):
                with open(folder+'/moppo_agents_warmup_'+str(i)+'.pickle', 'wb') as handle:
                    pickle.dump(agents[i].state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(folder+'/archive_data.pickle', 'wb') as handle:
                pickle.dump(archive_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for i in range(num_weight_vectors):
            with open(folder+'/moppo_agents_data_'+str(i)+'.pickle', 'wb') as handle:
                pickle.dump(agents[i].state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        print("Total samples mopg:",np.shape(a_arch_np)[0])
        print("Total samples moea:",np.shape(x_moea)[0])
        
        if n_iters == 0:
            x_dat = a_arch_np
            y_dat = ret_arch_np
            p_buf_start = p_buf_agents
        else:
            x_dat = np.vstack((a_arch_np,x_moea))
            y_dat = np.vstack((ret_arch_np,y_moea))
            p_buf_start = np.vstack((p_buf_agents,p_buf_moea_agents))
        
        dat_index_temp = None
        for i in range(num_weight_vectors):
            if dat_index_temp is None:
                dat_index_temp = np.arange(args.nupdates_warmup*(i+1)-num_updates,args.nupdates_warmup*(i+1))
            else:
                dat_index_temp =np.hstack((dat_index_temp,np.arange(args.nupdates_warmup*(i+1)-num_updates,args.nupdates_warmup*(i+1))))
        if n_iters>0:
            dat_index_temp = np.hstack((dat_index_temp,np.arange(args.nupdates_warmup*num_weight_vectors,np.shape(x_dat)[0])))
        
        print("Total samples for moea:",np.shape(x_dat[dat_index_temp])[0])
        print("Init pop size:", np.shape(p_buf_start)[0])
        if n_iters==0:
            x_z = p_buf_agents
            y_z = p_buf_ret
        else:
            x_z = np.vstack((p_buf_agents,p_buf_moea_agents))
            y_z = np.vstack((p_buf_ret,p_buf_moea_ret))
        
        # Run surrogate assisted MOEA
        population = run_optimizer(num_objs,
                                      num_vars,
                                      #x_dat[dat_index_temp],
                                      #y_dat[dat_index_temp],
                                      x_z,
                                      y_z,
                                      p_buf_start
                                      )
        indivs = population.individuals
        solns = population.objectives
        num_indiv = indivs.shape[0]
        
        # Evaluate solutions of MOEA

        evaluated_objs = evaluate_vector(env_eval, indivs, agent_base, num_objs, mean_params, device)
        if n_iters == 0:
            x_moea = indivs
            y_moea = evaluated_objs
            p_buf_moea_ret = evaluated_objs
            p_buf_moea_agents = indivs
            nds_moea_agents = indivs
            nds_moea_ret = evaluated_objs
        else:
            x_moea = np.vstack((x_moea, indivs))
            y_moea = np.vstack((y_moea, evaluated_objs))
            p_buf_moea_agents= np.vstack((p_buf_moea_agents,indivs))
            p_buf_moea_ret = np.vstack((p_buf_moea_ret,evaluated_objs))
            nds_moea_agents = np.vstack((nds_moea_agents,indivs))
            nds_moea_ret = np.vstack((nds_moea_ret,evaluated_objs))
            
        performance_buffer_moea = APD_performance.APD_performance(p_buf_moea_ret,
                                                ref_vectors_performance,
                                                #(n_iters+1)/max_iters,
                                                0.001,
                                                num_objs,
                                                refx,
                                                )   
        if nds_moea_ret.shape[0]>1:
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = -nds_moea_ret)
        else:
            ndf=[0]
        p_buf_moea_agents = p_buf_moea_agents[performance_buffer_moea]
        p_buf_moea_ret = p_buf_moea_ret[performance_buffer_moea]
        if p_buf_moea_ret.ndim==1:
            p_buf_moea_ret = p_buf_moea_ret.reshape(1,-1)
            p_buf_moea_agents = p_buf_moea_agents.reshape(1,-1)  
        nds_moea_agents = nds_moea_agents[ndf[0]]
        nds_moea_ret = nds_moea_ret[ndf[0]]
        print("perf_mopg=",np.shape(p_buf_ret))
        print("perf_moea=",np.shape(p_buf_moea_ret))
        print("nds_mopg=",np.shape(nds_ret))
        print("nds_moea=",np.shape(nds_moea_ret))

        archive_data_moea={"x_moea":x_moea, 
                           "y_moea":y_moea,
                           "p_buf_moea_ret":p_buf_moea_ret,
                           "p_buf_moea_agents":p_buf_moea_agents}
        with open(folder+'/archive_data_moea.pickle', 'wb') as handle:
                pickle.dump(archive_data_moea, handle, protocol=pickle.HIGHEST_PROTOCOL)

        archive_data_iteration[str(n_iters)] = {"p_buf_agents":p_buf_agents,
                                "p_buf_critics":p_buf_critics,
                                "p_buf_ret":p_buf_ret,
                                "p_buf_weights":p_buf_weights,
                                "nds_agents":nds_agents,
                                "nds_ret":nds_ret,
                                "p_buf_moea_ret":p_buf_moea_ret,
                                "p_buf_moea_agents":p_buf_moea_agents,
                                "nds_moea_agents":nds_moea_agents,
                                "nds_moea_ret":nds_moea_ret}
        with open(folder+'/archive_data_iteration.pickle', 'wb') as handle:
            pickle.dump(archive_data_iteration, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #chosen_indiv = np.random.choice(num_indiv,num_weight_vectors)

        
        for i in range(num_weight_vectors):
            with open(folder+'/moppo_agents_data_'+str(i)+'.pickle', 'wb') as handle:
                pickle.dump(agents[i].state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)


        #### Adapt worker weights and performance buffer weights
        weight_vect_obj.adapt(np.vstack((p_buf_ret,p_buf_moea_ret)))
        weight_vectors = np.flip(weight_vect_obj.values_planar,(0,1))
        weight_vect_perf.adapt(np.vstack((p_buf_ret,p_buf_moea_ret)))
        #ref_vectors_performance = np.flip(weight_vect_perf.values,(0,1))
        ref_vectors_performance = weight_vect_perf.values
        refx = weight_vect_perf.neighbouring_angles_current


        #### Load agents for performance buffer that matches the weight vector
        
        # ASF scalarize
        ideal_p_buf = np.max(p_buf_ret, axis=0)
        nadir_p_buf = np.min(p_buf_ret, axis=0)
        scale = ideal_p_buf - nadir_p_buf
        inner_term = (ideal_p_buf - p_buf_ret)/scale
        rho = 1e-6
        sum_term = rho * np.sum(rho*inner_term, axis=1)
        #w1 = np.reshape(weight_vectors,(-1,1,num_objs))
        #w2 = np.tile(w1,[np.shape(p_buf_ret)[0],1])
        p_buf_ret_scalar = np.zeros((np.shape(p_buf_ret)[0], num_weight_vectors))
        for wz in range(num_weight_vectors):     
            p_buf_ret_scalar[:,wz] = np.max(np.multiply(inner_term,weight_vectors[wz]),axis=1)
        argmax_ret = p_buf_ret_scalar.argmin(axis=0)
        
        #max_term = np.max(np.multiply(np.tile(((p_buf_ret-ideal_p_buf)/scale),[num_weight_vectors,1]),np.transpose(weight_vectors)))
        
        # For WS scalarization
        #p_buf_ret_scalar = np.dot(p_buf_ret,np.transpose(weight_vectors))
        #argmax_ret = p_buf_ret_scalar.argmax(axis=0)


        for wz in range(num_weight_vectors):
            agents[wz].actor_mean.load_state_dict(nparray_to_param(mean_params,p_buf_agents[argmax_ret[wz]],device))
            agents[wz].critic.load_state_dict(nparray_to_param(mean_params_critic,p_buf_critics[argmax_ret[wz]],device))


        fig = plt.figure()
        plt.scatter(-population.objectives[:,0], -population.objectives[:,1]) 
        plt.scatter(ret_arch_np[:,0], ret_arch_np[:,1], marker='^')
        #if n_iters != 0:
        plt.scatter(y_moea[:,0],y_moea[:,1],marker='^')
        plt.scatter(evaluated_objs[:,0], evaluated_objs[:,1])        
        plt.scatter(p_buf_moea_ret[:,0],p_buf_moea_ret[:,1],marker='*')
        plt.scatter(p_buf_ret[:,0],p_buf_ret[:,1],marker='*')
        plt.scatter(p_buf_ret[argmax_ret,0],p_buf_ret[argmax_ret,1],marker='*',c='black')
        plt.savefig(folder+'/plot_objs_'+str(n_iters)+'.pdf')

        print("**************** Global Steps ***** == ",global_step)
        n_iters +=1



    envs.close()
    writer.close()
                    


        
        





