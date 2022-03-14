import gym
import gym_nmf
from stable_baselines3 import PPO
import stable_baselines3.common.logger

env = gym.make('nmf18-pos2pos_distance-v0',
               state_indices=[-100, -75, -25, 0],
               run_time=1, time_step=5e-4, kp=0.1,
               headless=False, with_ball=True)
logger = stable_baselines3.common.logger.configure(
    './log/run2', ['stdout', 'csv', 'tensorboard'])
# obs = env.reset()

model = PPO('MlpPolicy', env, verbose=1)
model.set_logger(logger)
model.learn(total_timesteps=2000*1000)
model.save('run2')
print('done')

# obs = env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     # env.render()
#     if done:
#       obs = env.reset()