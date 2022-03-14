from gym.envs.registration import register
from importlib_metadata import entry_points

register(
    id='nmf18-simple_position_control-v0',
    entry_point='nmf_gym.envs:NMF18SimplePositionControlEnv'
)

register(
    id='nmf18-pos2pos_distance-v0',
    entry_point='nmf_gym.envs:NMF18Pos2PosDistanceEnv'
)
