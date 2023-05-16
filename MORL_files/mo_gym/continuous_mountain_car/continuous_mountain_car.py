import math

import numpy as np
from gym import spaces
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

class Continuous_MOMountainCar(Continuous_MountainCarEnv):

    def __init__(self, goal_velocity=0, render_mode=None):
        super().__init__(goal_velocity)

        self.reward_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.render_mode  =render_mode
        self.done = False
        #self.max_
    
    #def _is_truncated(self) -> bool:
    #    """The episode is over if the ego vehicle crashed or the time is out."""
    #    return self.time >= self.config["duration"]

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
        #terminated = bool(
        #    position >= self.goal_position and velocity >= self.goal_velocity
        #)
        
        """
        reward = np.zeros(3, dtype=np.float32)
        if terminated:
            reward[0] = 0
        else:
            reward[0] = -1
        
        if action[0] < 0:
            reward[1] -= -math.pow(action[0], 2) * 0.1
        elif action[0] > 0:
            reward[2] -= math.pow(action[0], 2) * 0.1
        else:
            reward[1] = 0
            reward[2] = 0
        """
        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        #done = self.done
        if not self.done:
            self.done = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        #reward = -1.0
        #reward = np.zeros(3, dtype=np.float32)
        reward = np.zeros(2, dtype=np.float32)
        
        reward[0] = 0.0 if done else -1.0        # time penalty
        #reward[0] = 0.0 if done else -np.abs(self.goal_position-position)        # time penalty
        
        
        #reward[1] = 0.0 if action[0] > 0 else -1.0 # reverse penalty
        #reward[2] = 0.0 if action[0] < 0 else -1.0 # forward penalty
        reward[1] = 0.0 if done else -np.abs(force) #energy penalty
        """

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        #reward = -1.0
        reward = np.zeros(2, dtype=np.float32)
        reward[0] = 0.0 if done else -1.0        # time penalty
        reward[1] = -np.abs(action[0]) # energy penalty
        """

        self.state = np.array([position, velocity], dtype=np.float32)
        
        if self.render_mode == 'human':
            self.render()
        """
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        #reward = -1.0
        reward = np.zeros(3, dtype=np.float32)
        reward[0] = 0.0 if done else -1.0        # time penalty
        reward[1] = 0.0 if action != 0 else -1.0 # reverse penalty
        reward[2] = 0.0 if action != 2 else -1.0 # forward penalty

        self.state = (position, velocity)
        """
        truncated = False

        return np.array(self.state, dtype=np.float32), 0, done, {'reward':reward}
