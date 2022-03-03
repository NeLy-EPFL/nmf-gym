from gym.envs.registration import register
from importlib_metadata import entry_points

register(
    id='nmf-v0',
    entry_point='gym_nmf.envs:NMF18PositionControlEnv'
)