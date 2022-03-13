import gym
import gym_nmf
from stable_baselines3 import A2C

env = gym.make('nmf18-pos2pos_distance-v0',
               state_indices=[-100, -75, -25, 0],
               run_time=1, time_step=5e-4, kp=0.1,
               headless=True, with_ball=True)

# obs = env.reset()

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()