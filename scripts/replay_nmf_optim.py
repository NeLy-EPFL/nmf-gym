"""
Replay the optimization result from NMF paper optimization using
position controllers with the Gym environment. Use this to determine
the upper limit of joint torques and as a sanity check.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
from enum import Enum

import nmf_gym
from nmf_gym.envs import NMF18SimplePositionControlEnv


class SaveMode(Enum):
    NO_SAVE = 0
    VIDEO = 1
    FRAMES = 2

TO_SAVE = SaveMode.VIDEO

if __name__ == '__main__':
    # Initialize gym env
    if TO_SAVE == SaveMode.VIDEO:
        env = NMF18SimplePositionControlEnv(
            run_time=6, time_step=5e-4, kp=0.4,
            headless=False, with_ball=True,
            movie_name='test_movie.mp4'
        )
    elif TO_SAVE == SaveMode.FRAMES:
        rec_options = {'save_frames': True}
        env = NMF18SimplePositionControlEnv(run_time=6, time_step=5e-4, kp=0.4,
                                            headless=False, with_ball=True,
                                            sim_options=rec_options)
    else:
        env = NMF18SimplePositionControlEnv(run_time=6, time_step=5e-4, kp=0.4,
                                            headless=False, with_ball=True)
    
    # Load target position dataframe
    tgt_joint_pos_path = (
        Path(nmf_gym.__path__[0]).parent / 
        'data/nmf_paper_optim_replay/joint_positions_from_paper.h5'
    )
    tgt_joint_pos_df = pd.read_hdf(tgt_joint_pos_path)
    tgt_joint_pos_df = tgt_joint_pos_df[env.act_joints]
    
    # Run simulation
    obs_hist = []
    reward_hist = []
    for i in trange(env.max_niters):
        obs, reward, is_done, _ = env.step(tgt_joint_pos_df.iloc[i])
        assert is_done == (i == env.max_niters - 1)
        obs_hist.append(obs)
        reward_hist.append(reward)
    env.close()
    
    # Plot joint positions
    t_grid = np.arange(env.max_niters) * env.time_step
    for ij, joint in enumerate(env.act_joints):
        if joint.startswith('joint_L'):
            continue
        pos_ts = [obs[ij] for obs in obs_hist]
        plt.plot(t_grid, np.rad2deg(pos_ts))
    plt.title('Joint Positions')
    plt.xlabel('Time (s)')
    plt.ylabel('Position ($\degree$)')
    plt.tight_layout()
    plt.show()

    # Plot joint velocities
    t_grid = np.arange(env.max_niters) * env.time_step
    for ij, joint in enumerate(env.act_joints):
        if joint.startswith('joint_L'):
            continue
        vel_ts = [obs[len(env.act_joints) + ij] for obs in obs_hist]
        plt.plot(t_grid, np.rad2deg(vel_ts))
    plt.title('Joint Velocities')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity ($\degree$/s)')
    plt.tight_layout()
    plt.show()

    # Plot joint torques
    t_grid = np.arange(env.max_niters) * env.time_step
    for ij, joint in enumerate(env.act_joints):
        if joint.startswith('joint_L'):
            continue
        torq_ts = [obs[len(env.act_joints) * 2 + ij] for obs in obs_hist]
        plt.plot(t_grid, torq_ts)
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
