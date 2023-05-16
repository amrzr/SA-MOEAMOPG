from typing import Dict, Text, Optional

import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.envs.common.action import Action

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import yaml
import os
import pathlib
from typing import List, Tuple, Optional, Callable, TypeVar, Generic, Union, Dict, Text

Observation = np.ndarray

class CustomTwoWayEnv(AbstractEnv):

    # load driver types configuration
    file_dir = pathlib.Path(__file__).parent.resolve()
    conf_dir = os.path.join(file_dir.parent.resolve(), "conf/NPC.yaml")
    with open(conf_dir, "r") as f:
        yamlconfig = yaml.load(f, Loader=yaml.FullLoader)

    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        super().__init__(config, render_mode)


    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "collision_reward": -1,
            "high_speed_reward": 0.8,
            "NPC_interval": 1e2,
            "road_length" : 1e4,
        })
        return config

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        return sum(self.config.get(name, 0) * reward for name, reward in self._rewards(action).items())

    def _rewards(self, action: int) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        return {
            "high_speed_reward": self.vehicle.speed_index / (self.vehicle.target_speeds.size - 1),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> np.ndarray:
        if self.config.get("lanes_count")%2 != 0:
            exit("\033[1;33m Warning: \033[0m lanes_count must be even number in this environment!")
        self._make_road()
        self._make_vehicles()

    def _make_road(self, start=0,angle=0,speed_limit=50):
        """
        Make a road composed of a two-way road.

        :return: the road
        """
        lanes_count = self.config["lanes_count"]
        self.road_length = self.config["road_length"]
        self.NPC_interval = self.config["NPC_interval"]
        net = RoadNetwork()

        # add forward lanes
        nodes_str = ("a", "b")
        for lane in range(lanes_count):
            origin = np.array([start, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([start + self.road_length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            origin = rotation @ origin
            end = rotation @ end
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes_count - 1 else LineType.NONE]
            net.add_lane(*nodes_str, StraightLane(origin, end, line_types=line_types, speed_limit=speed_limit))

        # add backward lanes
        nodes_str = ("b", "a")
        for lane in range(lanes_count):
            origin = np.array([start + self.road_length, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([start, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            origin = rotation @ origin
            end = rotation @ end
            line_types = [LineType.CONTINUOUS_LINE if lane == int(lanes_count/2)-1 else None,
                          LineType.NONE]
            net.add_lane(*nodes_str, StraightLane(origin, end, line_types=line_types, speed_limit=speed_limit))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the road

        :return: the ego-vehicle
        """

        lanes_count = self.config["lanes_count"]
        vehicles_count = int(self.config["road_length"]/self.config["NPC_interval"])*2

        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("a", "b", lanes_count-1)).position(30, 0),
                                                     speed=30)
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        half_lanes_count = int(lanes_count/2)

        for i in range(int(vehicles_count/2)):
            # Driver types Proportion
            driver_type_list = []
            Proportion_list = []
            for _, key in enumerate(list(self.yamlconfig.keys())):
                driver_type_list.append(key)
                Proportion_list.append(self.yamlconfig[key]['Proportion'])
            Px = np.array(Proportion_list)
            driver_type = np.random.choice(driver_type_list, p=Px/Px.sum())

            # IDM_MOBIL_parameters
            target_speed = self.yamlconfig[driver_type]['target_speed']
            ACC_MAX = self.yamlconfig[driver_type]['ACC_MAX']
            COMFORT_ACC_MAX = self.yamlconfig[driver_type]['COMFORT_ACC_MAX']
            COMFORT_ACC_MIN = self.yamlconfig[driver_type]['COMFORT_ACC_MIN']
            DISTANCE_WANTED = self.yamlconfig[driver_type]['DISTANCE_WANTED']
            TIME_WANTED = self.yamlconfig[driver_type]['TIME_WANTED']
            DELTA = self.yamlconfig[driver_type]['DELTA']
            DELTA_RANGE = self.yamlconfig[driver_type]['DELTA_RANGE']
            POLITENESS = self.yamlconfig[driver_type]['POLITENESS']
            LANE_CHANGE_MIN_ACC_GAIN = self.yamlconfig[driver_type]['LANE_CHANGE_MIN_ACC_GAIN']
            LANE_CHANGE_MAX_BRAKING_IMPOSED = self.yamlconfig[driver_type]['LANE_CHANGE_MAX_BRAKING_IMPOSED']
            LANE_CHANGE_DELAY = self.yamlconfig[driver_type]['LANE_CHANGE_DELAY']
            IDM_MOBIL_parameters = [target_speed, ACC_MAX, COMFORT_ACC_MAX, COMFORT_ACC_MIN, DISTANCE_WANTED, TIME_WANTED, DELTA, DELTA_RANGE, POLITENESS, LANE_CHANGE_MIN_ACC_GAIN, LANE_CHANGE_MAX_BRAKING_IMPOSED, LANE_CHANGE_DELAY]
            
            # Vehicles size
            vehicle_size = self.yamlconfig[driver_type]['Vehicle_size']

            # forward NPC
            spawn_lane = np.random.choice(half_lanes_count)+half_lanes_count
            self.road.vehicles.append(
                vehicles_type(road,
                            position=road.network.get_lane(("a", "b", spawn_lane))
                            .position(70 + self.NPC_interval*i, 0),
                            heading=road.network.get_lane(("a", "b", spawn_lane)).heading_at(70+self.NPC_interval*i),
                            speed=IDM_MOBIL_parameters[0],
                            target_speed = IDM_MOBIL_parameters[0],
                            enable_lane_change=True,
                            IDM_MOBIL_parameters=IDM_MOBIL_parameters,
                            vehicle_size=vehicle_size)
            )
            
            # backward NPC
            spawn_lane = np.random.choice(half_lanes_count)
            self.road.vehicles.append(
                vehicles_type(road,
                            position=road.network.get_lane(("b", "a", spawn_lane))
                            .position(self.road_length-70-self.NPC_interval*i, 0),
                            heading=road.network.get_lane(("b", "a", spawn_lane)).heading_at(self.road_length-70-self.NPC_interval*i),
                            speed=IDM_MOBIL_parameters[0],
                                target_speed = IDM_MOBIL_parameters[0],
                            enable_lane_change=True,
                            IDM_MOBIL_parameters=IDM_MOBIL_parameters,
                            vehicle_size=vehicle_size)
            )

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

        return obs, reward[0], terminated, {'reward':vec_reward}
register(
    id='custom-two-way-v0',
    entry_point='highway_env.envs:CustomTwoWayEnv',
    max_episode_steps=9999
)
