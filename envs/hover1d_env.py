import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Physics, rules, and scoring part

class Hover1DEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Physical constants
        # sized (roughly) for irl drone and Crazyfile
        self.mass = 0.03        # kg
        self.gravity = 9.81     # m/s^2
        self.dt = 0.02          # s per step (~50 Hz)
        self.max_thrust = 2 * self.mass * self.gravity # lifts about 2x its weight
        self.target_z = 1.0     # m
        self.max_steps = 500    # 10 sec episodes
        self.abonus = 1.0       # added bonus for staying alive

        self.action_space = spaces.Box(low= -1.0, high= 1.0, shape= (1,), dtype= np.float32)
        # what it can do/outputs (driving terms like steering/gas/break)

        self.observation_space = spaces.Box(low= -np.inf, high= np.inf, shape= (2,), dtype= np.float32)
        # what it can know/input (driving terms like speed/road/mirrors)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.z = self.np_random.uniform(0.2, 1.8) # random start height (meters)
        self.vz = self.np_random.uniform(-0.5, 0.5) # random start velocity (m/s)
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        # arrives as (1,) we want the number
        a = float(action[0])

        # mapping [-1,1] tp [0, 1] to [0, max thrust]
        thrust = (a + 1) / 2*self.max_thrust

        acc = thrust / self.mass - self.gravity # net accel
        self.vz = self.vz + acc * self.dt # velocity
        self.z = self.z + self.vz * self.dt # position

        # reward
        error = self.z - self.target_z
        reward = self.abonus -abs(error)

        # terminated
        terminated = (self.z > 3.0) or (self.z < 0.1)

        self.steps += 1
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated,truncated, {}

    def _get_obs(self):
        return np.array([self.z - self.target_z, self.vz], dtype=np.float32)