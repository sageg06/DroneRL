from stable_baselines3.common.env_checker import check_env
from envs.hover1d_env import Hover1DEnv

check_env(Hover1DEnv())   # yells at you if shapes/types are wrong
print("env OK")