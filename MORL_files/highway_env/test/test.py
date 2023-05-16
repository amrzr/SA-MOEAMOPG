import gym
import os, sys
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(1, os.path.abspath("../.."))
sys.path.append('/home/atanu_aalto/work/codes/RLscratch/')
import highway_env
import math
import numpy as np
import argparse
#from ilqr.arguments import add_arguments
#argparser = argparse.ArgumentParser(description='CILQR')
#add_arguments(argparser)
#args = argparser.parse_args()
np.random.seed(999)

env = gym.make('custom-two-way-v0')
env.configure({
    "observation": {
        "vehicles_count": 4, # including ego
        "features": ["presence", "x", "y", "vx", "vy", "heading"],
        "type": "Kinematics",
        "normalize": False,
        "absolute": True,
    },
    "action": {  
        "type": "ContinuousAction",
        "ACCELERATION_RANGE": 50,
        "STEERING_RANGE": 50,
        "speed_range": 10,
    },
    "manual_control": True,
    "lanes_count":10,
    "duration": 1000,  # [s]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 1800,  # [px]
    "screen_height": 450,  # [px]
    "forward_npc_interval": 10,
    "backward_npc_interval": 10,
    "road_length" : 100,
    "road_speed_limit" : 10, # m/s
    "simulation_frequency": 15, # for intermdiate simulation frames rendering
    "policy_frequency": 15, 
    # reward
    "high_speed_reward": 0.8,
})


done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)  # if manual control is true, these actions are ignored
    # print("action:  ", action)
    # print("obs: ", obs)
    # print("reward: ", reward)
    # print("done: ", done)
    # print("info: ", info)

    env.render()

# obs_array_max = np.max(obs_array, axis=0)
# print("obs_array_max:", obs_array_max)

# obs_array_min = np.min(obs_array, axis=0)
# print("obs_array_min:", obs_array_min)

# "features": ["presence", "x", "y", "vx", "vy", "heading"],
# "processed features for ego and Npcs": ["1", "x/args.road_length", "(y-downLane)/(upLane-downLane)", "vx/args.max_speed", "vy/max_speed", "heading%pi/pi"],
# obs_array_max: [1 1.3430857e+03 2.3641930e+01 5.9999615e+01 5.9995399e+01 6.5523901e+00]
# obs_array_min: [0.         0.       -50.674644 -59.994728 -60.006668 -17.302334]
