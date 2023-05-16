import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.join(base_dir, '/env_mujoco/'))
import warnings
warnings.filterwarnings("ignore")
import random
import time
from distutils.util import strtobool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import mo_utils
import pickle


def scalarize_ASF_tensor(fitness_vector, 
                weights, 
                nadir_point,
                utopian_point,
                num_envs,
                device,
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
        #if x_slice_end-x_slice_start > 1:
        x_slice = np.reshape(x[x_slice_start:x_slice_end],param.size())
        #else:
        #    x_slice = x[x_slice_start:x_slice_end]
        mean_params[name] = torch.Tensor(x_slice).cuda()
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

def run_mopg(agent, weights, num_updates, envs, env_eval, global_step, device, tot_steps, max_updates, args):
    tens_weights = torch.Tensor(weights.copy()).to(device)
    weights2=np.array([weights])
    weights_vect_repeat = np.repeat(weights2,np.full((1),args.num_steps),axis=0)
    #weights_vect_tiled = np.tile(weights_vect_repeat,(1,args.num_envs,1))
    w2 = np.reshape(weights_vect_repeat,(-1,1,args.num_objs))
    w3 = np.tile(w2,[args.num_envs,1])
    w_tens = torch.Tensor(w3.copy()).to(device)
    b_weights = w_tens.reshape((-1,)+(args.num_objs,))

    agents_archive_mopg = []
    episodic_return_archive_mopg = []
    weights_archive_mopg = []
    critics_archive_mopg = []
    episodic_deter_return_archive = []
  
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_num_threads(2)
    #device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # env setup
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    # ALGO Logic: Storage setup
    num_objs = args.num_objs
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, num_objs, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, num_objs, args.num_envs)).to(device)    
    start_time = time.time()
    next_obs = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    #num_updates = args.total_timesteps // args.batch_size
    
    mopg_step = 0
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    next_obs = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    for update in range(1, num_updates + 1):
        print("**** Weights **** :",weights)
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            #frac = 1.0 - (mopg_step - 1.0) / (args.num_steps*num_updates)
            frac = 1.0 - (mopg_step+tot_steps - 1.0) / (args.num_steps*max_updates*args.num_envs)
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            print("Frac:",frac)
            print("LR:",lrnow)
        tot_rew=np.zeros(args.num_objs)
        t_count = 0
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            mopg_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, tens_weights.tile([next_obs.shape[0],1])) 
                values[step] = value.transpose(0,1)
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, _, terminated, infos = envs.step(action.cpu().numpy())
            reward = infos['reward']
            tot_rew=tot_rew+infos['reward'][0]
            reward = np.vstack(reward[:]).astype(np.float32)
            done = terminated
            if np.any(terminated):
                t_count += 1
            rewards[step] = torch.tensor(reward.transpose()).to(device)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        episodic_total_return = rewards.sum(2).sum(0)/(torch.sum(dones) + torch.sum(1-next_done))
        
        print("exp_ep_rew",episodic_total_return)
        print("t_count:",t_count)
        print("MOPG_step:",mopg_step)
        
       
        # bootstrap value if not done
        with torch.no_grad():
            #next_value = agent.get_value(next_obs).reshape(1, -1)
            next_value = agent.get_value(next_obs, tens_weights.tile([next_obs.shape[0],1])).transpose(0,1).reshape(num_objs, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

            ##### scalarizing advantages
            nadir = advantages.min(2)[0].min(0)[0]
            ideal = advantages.max(2)[0].max(0)[0]
            advantages = mo_utils.scalarize_WS_tensor(advantages, weights, args.num_envs, device)
            #advantages = scalarize_ASF_tensor(advantages, weights, nadir, ideal, args.num_envs, device)


        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape((-1,) + (args.num_objs,))
        b_values = values.reshape((-1,) + (args.num_objs,))
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_weights[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if args.clip_vloss:
                    v_loss_unclipped = (torch.linalg.norm((newvalue - b_returns[mb_inds]),dim=1)) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (torch.linalg.norm((newvalue - b_returns[mb_inds]),dim=1)) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((torch.linalg.norm((newvalue - b_returns[mb_inds]),dim=1)) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        #print("Global Steps :", global_step)

        episodic_deter_return = collect_rollouts(env_eval, agent, device, args)
        print("Deterministic return:",episodic_deter_return)
        mean_params = dict((key, value) for key, value in agent.actor_mean.state_dict().items())
        mean_params_critic = dict((key, value) for key, value in agent.critic.state_dict().items())
        params_size = 0
        for name, param in mean_params.items():
            params_size += param.nelement()
        params_size_critic = 0
        for name, param in mean_params_critic.items():
            params_size_critic += param.nelement()  
        agent_vars = param_to_nparray(mean_params,params_size)
        critic_vars = param_to_nparray(mean_params_critic, params_size_critic)
        #agent_vars = np.append(agent_vars,weights)
        
        if update == 1:
            agents_archive_mopg=agent_vars
            episodic_return_archive_mopg=episodic_total_return.cpu().detach().numpy()
            critics_archive_mopg=critic_vars
            episodic_deter_return_archive=episodic_deter_return
        else:
            agents_archive_mopg = np.vstack((agents_archive_mopg, agent_vars))
            episodic_deter_return_archive = np.vstack((episodic_deter_return_archive, episodic_deter_return))
            critics_archive_mopg = np.vstack((critics_archive_mopg,critic_vars))
            


    #ep_return_final = collect_rollouts(envs, agent, device, args)
    #return agent, ep_return_final, agent.critic, agents_archive_mopg, episodic_return_archive_mopg, critics_archive_mopg
    return agent, agents_archive_mopg, critics_archive_mopg, episodic_deter_return_archive



