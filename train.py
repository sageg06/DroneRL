from stable_baselines3 import PPO
from envs.hover1d_env import Hover1DEnv
env = Hover1DEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("1ddroneflight_ppo")