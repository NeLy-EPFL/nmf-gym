"""
Replay the optimization result from NMF paper optimization using
position controllers with the Gym environment. Use this to determine
the upper limit of joint torques and as a sanity check.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import gym_nmf
from gym_nmf.envs import NMF18PositionControlEnv


if __name__ == '__main__':
    # Initialize gym env
    env = NMF18PositionControlEnv(run_time=6, time_step=5e-4, kp=0.4,
                                  headless=False, with_ball=True)
    
    # Load target position dataframe
    tgt_joint_pos_path = (
        Path(gym_nmf.__path__[0]).parent / 
        'data/nmf_paper_optim_replay/joint_positions_from_paper.h5'
    )
    tgt_joint_pos_df = pd.read_hdf(tgt_joint_pos_path)
    tgt_joint_pos_df = tgt_joint_pos_df[env.act_joints]
    
    # Run simulation
    obs_hist = []
    reward_hist = []
    for i in range(env.max_niters):
        obs, reward, is_done, _ = env.step(tgt_joint_pos_df.iloc[i])
        assert is_done == (i == env.max_niters - 1)
        obs_hist.append(obs)
        reward_hist.append(reward)
    
    # Plot joint torques
    t_grid = np.arange(env.max_niters) * env.time_step
    for ij, joint in enumerate(env.act_joints):
        if joint.startswith('joint_L'):
            continue
        pos_ts = [obs[len(env.act_joints) * 2 + ij] for obs in obs_hist]
        plt.plot(t_grid, pos_ts)
    plt.title('Applied Joint Torques')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (PyBullet unitless value)')
    plt.tight_layout()
    plt.show()

    # Plot base position
    plt.plot([obs[-6] for obs in obs_hist], [obs[-5] for obs in obs_hist])
    plt.title('Fly Base Position')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.tight_layout()
    plt.show()