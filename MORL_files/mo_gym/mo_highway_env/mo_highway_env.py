import math

import numpy as np
from gym import spaces
#from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
#from highway_env.envs.highway_env import HighwayEnvMO
from gym import RewardWrapper

class MOHighwayEnv(RewardWrapper):

    def __init__(self):
        super().__init__(goal_velocity)

        self.reward_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    
    def step(self, action: int):
        
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        # Convert a possible numpy bool to a Python bool.
        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        
        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        #reward = -1.0
        #reward = np.zeros(3, dtype=np.float32)
        reward = np.zeros(2, dtype=np.float32)
        reward[0] = 0.0 if done else -1.0        # time penalty
        
        #reward[1] = 0.0 if action[0] > 0 else -1.0 # reverse penalty
        #reward[2] = 0.0 if action[0] < 0 else -1.0 # forward penalty
        reward[1] = -np.abs(force) #energy penalty


        self.state = np.array([position, velocity], dtype=np.float32)
        

        return np.array(self.state, dtype=np.float32), reward, terminated, {}
