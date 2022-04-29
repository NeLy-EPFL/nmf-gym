from gym.envs.registration import register
from importlib_metadata import entry_points

register(
    id='nmf-simple_position_control-v0',
    entry_point='nmf_gym.envs:NMFSimplePositionControlEnv'
)

register(
    id='nmf-pos2pos_distance-v0',
    entry_point='nmf_gym.envs:NMFPos2PosDistanceEnv'
)
