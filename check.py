from stable_baselines3.common.env_checker import check_env
from envs.hover2d_env import Hover2DEnv

check_env(Hover2DEnv())   # yells at you if shapes/types are wrong
print("env OK")