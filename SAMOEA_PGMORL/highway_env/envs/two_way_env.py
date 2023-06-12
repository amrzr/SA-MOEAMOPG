from typing import Dict, Text

import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from typing import List, Tuple, Optional, Callable, TypeVar, Generic, Union, Dict, Text
from highway_env.envs.common.action import Action
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from gym import spaces

Observation = np.ndarray

class TwoWayEnv(AbstractEnv):

    """
    A risk management task: the agent is driving on a two-way lane with icoming traffic.

    It must balance making progress by overtaking and ensuring safety.

    These conflicting objectives are implemented by a reward signal and a constraint signal,
    in the CMDP/BMDP framework.
    """
    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        super().__init__(config, render_mode)

    #def __init__(self):
    #    super().__init__()
        self.reward_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
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
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,   # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 5,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 40],
            "co2_range": [0,500],
            "co2_reward":5,
            #"normalize_reward": True,
            "offroad_terminal": False,
            "left_lane_reward": 0.2,
            "left_lane_constraint": 1,
            'offroad_terminal': True,
            "simulation_frequency": 60,
            "policy_frequency": 60,
            "acceleration_reward": -1,
            "steering_reward": -0.0,
            "collision_reward": -10,
            "on_road_reward": -10
        })
        return config
    
    """
    
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "TimeToCollision",
                "horizon": 5
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "collision_reward": 0,
            "left_lane_constraint": 1,
            "left_lane_reward": 0.2,
            "high_speed_reward": 0.8,
        })
        return config
    
    def _reward(self, action: int) -> float:
        
        The vehicle is rewarded for driving with high speed
        :param action: the action performed
        :return: the reward of the state-action transition
        
        return sum(self.config.get(name, 0) * reward for name, reward in self._rewards(action).items())

    def _rewards(self, action: int) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        return {
            "high_speed_reward": self.vehicle.speed_index / (self.vehicle.target_speeds.size - 1),
            "left_lane_reward": (len(neighbours) - 1 - self.vehicle.target_lane_index[2]) / (len(neighbours) - 1),
        }
    """
    def co2_emissions(self):
        Gamma_idl = 8887 / 1000
        Eta = 1
        Epsilon_gas = 33.7
        r = 0
        C_rr = 0.015
        C_da = 0.079
        rho = 1.225
        M = 1678
        v = self.vehicle.speed
        a = self.vehicle.action['acceleration'] 
        g = 9.8

        f_RL = M*a + M*g*C_rr + 0.5*C_da*rho*v**2
        if f_RL >= 0:
            zeta = (Gamma_idl/(Epsilon_gas*Eta))*f_RL
        else: 
            zeta = ((r*Gamma_idl)/(Epsilon_gas*Eta))*f_RL

        return zeta

    # Conlon, J., & Lin, J. (2019). Greenhouse gas emission impact of autonomous
    #  vehicle introduction in an urban network. Transportation Research Record, 
    #  2673(5), 142-152. doi: 10.1177/0361198119839970
    def co2_emissions_2(self,type='light_passenger',fuel='gasoline'):
        
        acceleration = self.vehicle.action['acceleration'] 
        velocity = self.vehicle.speed

        if fuel == 'gasoline':
            T_idle = 2392    # CO2 emission from gasoline [gCO2/L]
            E_gas =  31.5e6  # Energy in gasoline [J\L]
        elif fuel == 'diesel':
            T_idle = 2660   # CO2 emission from diesel [gCO2/L]
            E_gas =  38e6   # Energy in diesel [J\L]

        if type == 'light_passenger':
            M = 1334    # light-duty passenger vehicle mass [kg]
        elif type == 'light_van':
            M = 1752    # light-duty van vehicle mass [kg]
        
        Crr = 0.015     # Rolling resistance
        Cd  = 0.3       # Aerodynamic drag coefficient
        A = 2.5         # Frontal area [m2]
        g = 9.81        # Gravitational acceleration
        r = 0           # Regeneration efficiency ratio
        pho = 1.225     # Air density
        fuel_eff = 0.7  # fuel efficiency [70%]

        
        condition = M  * acceleration * velocity + M  * g * Crr * velocity +0.5 * Cd * A  * pho * velocity **3
        
        Ei = T_idle  / E_gas  * condition

        if Ei <= 0:
            E = r
        else:
            Ei = Ei * (velocity + 0.5 * acceleration)
            E = Ei/fuel_eff

        return np.abs(E)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        reward = np.zeros(2, dtype=np.float32)
        rewards = self._rewards(action)        
        reward[0] = sum(self.config.get(name, 0) * reward_temp for name, reward_temp in rewards.items())
        
        #if self.config["normalize_reward"]:
        #    reward[0] = utils.lmap(reward[0],
        #                        [self.config["collision_reward"],
        #                            self.config["high_speed_reward"] + self.config["right_lane_reward"]],
        #                        [0, 1])
        
        #if rewards['on_road_reward'] == 0:
        #    reward[0] = -1000
        #if self.vehicle.crashed:
        #    reward[0]=-1000
        #else:
        #    reward[0] *= rewards['on_road_reward']
        #reward[1] = -np.abs(rewards['energy'])

        #co2 = 5000 - self.co2_emissions()
        #scaled_co2 = utils.lmap(co2, self.config["co2_range"], [0, 1])

        co2 = -self.co2_emissions_2()
        scaled_co2 = utils.lmap(co2, self.config["co2_range"], [0, 1])
        
        reward[1] = 5+self.config["co2_reward"]*scaled_co2 +(self.config["collision_reward"]*rewards["collision_reward"]
                      +self.config["on_road_reward"]*rewards["on_road_reward"])
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        #print(action)
        #if type(action) != int and type(action) != np.int64:
        #    energy = action[0]
        #else:
        #    energy = action
        if type(action) != int and type(action) != np.int64:
            acc_rew = np.abs(action[0])
            st_rew = np.abs(action[1])
        else:
            acc_rew = action
            st_rew = action
        return {
            "collision_reward": 1 if self.vehicle.crashed else 0,
            #"right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": 1 if not self.vehicle.on_road else 0,
            #"energy": float(energy)
            #"acceleration_reward": acc_rew,
            "steering_reward": st_rew,
        }




    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> np.ndarray:
        self._make_road()
        self._make_vehicles()

    def _make_road(self, length=10000):
        """
        Make a road composed of a two-way road.

        :return: the road
        """
        net = RoadNetwork()

        # Lanes
        net.add_lane("a", "b", StraightLane([0, 0], [length, 0],
                                            line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED)))
        net.add_lane("a", "b", StraightLane([0, StraightLane.DEFAULT_WIDTH], [length, StraightLane.DEFAULT_WIDTH],
                                            line_types=(LineType.NONE, LineType.CONTINUOUS_LINE)))
        net.add_lane("b", "a", StraightLane([length, 0], [0, 0],
                                            line_types=(LineType.NONE, LineType.NONE)))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the road

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("a", "b", 1)).position(30, 0),
                                                     speed=30)
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for i in range(3):
            self.road.vehicles.append(
                vehicles_type(road,
                              position=road.network.get_lane(("a", "b", 1))
                              .position(70+40*i + 10*self.np_random.normal(), 0),
                              heading=road.network.get_lane(("a", "b", 1)).heading_at(70+40*i),
                              speed=24 + 2*self.np_random.normal(),
                              enable_lane_change=False)
            )
        for i in range(2):
            v = vehicles_type(road,
                              position=road.network.get_lane(("b", "a", 0))
                              .position(200+100*i + 10*self.np_random.normal(), 0),
                              heading=road.network.get_lane(("b", "a", 0)).heading_at(200+100*i),
                              speed=20 + 5*self.np_random.normal(),
                              enable_lane_change=False)
            v.target_lane_index = ("b", "a", 0)
            self.road.vehicles.append(v)

    def step(self, action: Action) -> Tuple[Observation, np.ndarray, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        #if truncated:
        #    terminated = True
        info = self._info(obs, action)
        vec_reward = np.array([reward[0], reward[1]], dtype=np.float32)

        return obs, reward[0], terminated, {'obj':vec_reward}
        #return obs, reward, terminated, info
    
    
    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
    ) -> Tuple[Observation, dict]:
        
        #Reset the environment to it's initial configuration

        #:return: the observation of the reset state
        
        self.update_metadata()
        self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        self.time = self.steps = 0
        self.done = False
        self._reset()
        self.define_spaces()  # Second, to link the obs and actions to the vehicles once the scene is created
        obs = self.observation_type.observe()
        info = self._info(obs, action=self.action_space.sample())
        return obs
    
register(
    id='two-way-v0',
    entry_point='highway_env.envs:TwoWayEnv',
    max_episode_steps=3000
)
