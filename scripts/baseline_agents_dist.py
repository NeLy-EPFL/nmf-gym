"""
Using stable-baselines3 (SB3), train baseline agent with PPO that
maximizes walking distance. No penalty applied since this is only
a test.
"""

import gym
import gym_nmf
from pathlib import Path
from datetime import datetime
from stable_baselines3 import PPO
import stable_baselines3.common.logger


if __name__ == '__main__':
    env = gym.make('nmf18-pos2pos_distance-v0',
                state_indices=[-100, -75, -25, 0],
                run_time=1, time_step=5e-4, kp=0.1,
                headless=False, with_ball=True)
    log_dir = Path(f'./log/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f'Log dir: {log_dir.absolute()}')
    logger = stable_baselines3.common.logger.configure(
        str(log_dir), ['stdout', 'csv', 'tensorboard'])

    model = PPO('MlpPolicy', env, verbose=1)
    model.set_logger(logger)
    model.learn(total_timesteps=env.max_niters * 1000)
    model.save(log_dir / 'trained_model')
    print('Training done')

    # Roll out
    # obs = env.reset()
    # for i in range(env.max_niters):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     # env.render()
    #     if done:
    #       obs = env.reset()