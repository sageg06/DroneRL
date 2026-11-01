import matplotlib.pyplot as plt
from envs.hover1d_env import Hover1DEnv
from datetime import datetime
from stable_baselines3 import PPO
import numpy as np


model = PPO.load("hover1d_rev2_ppo.zip")
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
env = Hover1DEnv()
obs, info = env.reset()
env.mass = 0.036
env.disturbance = -0.029

heights = [env.z]
total_reward = 0
times = [0.0]
offset = np.mean(heights[250:]) - env.target_z

for step_number in range(1, 501):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    heights.append(env.z)
    total_reward += reward
    times.append(step_number * env.dt) # in seconds

    if terminated or truncated:
        break
env.close()
print(f"Lasted {env.steps} steps ({times[-1]:.2f} sec), total score: {total_reward:.1f}")
print(f"Steady state offset: {offset * 1000:.1f} mm")
# Plotting
plt.plot(times, heights, label="drone height")
plt.axhline(y=env.target_z, color="k", linestyle="--", label="target")
plt.xlabel("time (s)")
plt.ylabel("height (m)")
plt.legend()
plt.savefig(f"plots/run_{stamp}.png")
plt.show()

