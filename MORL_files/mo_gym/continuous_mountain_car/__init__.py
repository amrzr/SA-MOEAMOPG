from gym.envs.registration import register

register(
    id='mo-mountaincarcontinuous-v0',
    entry_point='mo_gym.continuous_mountain_car.continuous_mountain_car:Continuous_MOMountainCar',
    max_episode_steps=200
)