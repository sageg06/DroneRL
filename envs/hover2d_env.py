import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Hover2DEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Physical constants
        # sized (roughly) for irl drone and Crazyfile
        self.nom_mass = 0.03        # kg
        self.gravity = 9.81     # m/s^2
        self.dt = 0.02          # s per step (~50 Hz)

        self.arm_length = 0.033 # 0.033 m or 33mm (measured roughly real drone arm length
        self.inertia = 1.4e-5 # kg-m^2
        self.max_thrust = 2 * self.nom_mass * self.gravity # lifts about 2x its weight
        self.target_z = 1.0     # m
        self.target_y = 0.0     # m
        self.target_theta = 0.0 # rad, 0 being no tilt
        self.target_omega = 0.0 # rad/s, no tilt speed
        self.max_steps = 500    # 10 sec episodes

        # Noise
        self.z_noise_std = 0.005 # 5 mm
        self.vz_noise_std = 0.03 # 3 cm/s
        self.y_noise_std = 0.005 # 5 mm
        self.vy_noise_std = 0.03 # 3 m/s tilt speed
        self.theta_noise_std = 0.005
        self.omega_noise_std = 0.03

        # Reward weights
        self.abonus = 1.0       # added bonus for staying alive
        self.w_height = 0.5     # weight for if height is off
        self.w_tilt = 0.1      # weight if tilt is too much
        self.w_LRpos = 0.5      # weight if left/right pos isn't right

        # For rewards
        # Care most abt tilt - if tilt too much then drone crashes quickly
        # care second most abt height since it can recover being too high/low up to a point
        # position matters less since too right/left can be corrected



        self.action_space = spaces.Box(low= -1.0, high= 1.0, shape= (2,), dtype= np.float32)
        # what it can do/outputs (driving terms like steering/gas/break)

        self.observation_space = spaces.Box(low= -np.inf, high= np.inf, shape= (6,), dtype= np.float32)
        # what it can know/input (driving terms like speed/road/mirrors)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Positions + velocities
        # note for future self: y is left/right z is up/down
        self.z = self.np_random.uniform(0.2, 1.8) # random start height (meters)
        self.vz = self.np_random.uniform(-0.5, 0.5) # random start velocity (m/s)
        self.y = self.np_random.uniform(-0.5, 0.5) # +/- 0.5m random start position
        self.vy = self.np_random.uniform(-0.5, 0.5) # +/- 0.5m/s random start velocity

        self.theta = self.np_random.uniform(-0.2, 0.2) # +/- 0.2 rad random tilt angle
        # note to self: for tilt angle theta=0 is flat
        self.omega = self.np_random.uniform(-0.5, 0.5) #+/- 0.5 red/s tilt velocity

        self.mass = self.np_random.uniform(0.024,0.036)
        weight = self.nom_mass * self.gravity
        self.disturbance = self.np_random.uniform(-0.1 * weight, 0.1 * weight)
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        aL = float(action[0]) # Left motor
        aR = float(action[1]) # right motor

        left_thrust = (aL + 1) / 2*(self.max_thrust/2)
        right_thrust = (aR + 1) / 2*(self.max_thrust/2)
        total_thrust = (left_thrust + right_thrust)
        torque = self.arm_length * (right_thrust - left_thrust)
        # theta > 0 means RIGHT SIDE UP TILTED TOWARDS -y DO NOT FORGET!!!!!

        # Acceleration calculations
        a_y = (-total_thrust * np.sin(self.theta)) / self.mass
        a_z = (total_thrust * np.cos(self.theta) + self.disturbance) / self.mass - self.gravity
        alpha = torque / self.inertia # angular acceleration

        self.vz = self.vz + a_z * self.dt # velocity up/down
        self.z = self.z + self.vz * self.dt # position up/down

        self.vy = self.vy + a_y * self.dt # left/right velocity
        self.y = self.y + self.vy * self.dt # left/right position

        self.omega = self.omega + alpha * self.dt # tilt velocity
        self.theta = self.theta + self.omega * self.dt # tilt position

        # reward
        z_error = self.z - self.target_z
        y_error = self.y - self.target_y
        theta_error = self.theta - self.target_theta

        reward = self.abonus - (self.w_height * abs(z_error) + self.w_LRpos * abs(y_error)
                                + self.w_tilt * abs(theta_error))

        # terminated conditions
        term_high = (self.z > 3.0) or (self.z < 0.1)
        term_LRwall = (self.y > 3.0) or (self.y < -3.0)
        term_tilt = (self.theta > 1.0) or (self.theta < -1.0)
        terminated = term_high or term_LRwall or term_tilt

        self.steps += 1
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated,truncated, {}

    def _get_obs(self):
        # Height measurements
        zError_meas = (self.z - self.target_z) + self.np_random.normal(0.0, self.z_noise_std)
        vz_meas = (self.vz) + self.np_random.normal(0.0, self.vz_noise_std)

        # Left/Right measurements
        yError_meas = (self.y - self.target_y) + self.np_random.normal(0.0, self.y_noise_std)
        vy_meas = (self.vy) + self.np_random.normal(0.0, self.vy_noise_std)

        # tilt measurements
        thetaError_meas = (self.theta - self.target_theta) + self.np_random.normal(0.0, self.theta_noise_std)
        omega_meas = (self.omega - self.target_omega) + self.np_random.normal(0.0, self.omega_noise_std)

        return np.array([zError_meas, vz_meas, yError_meas,
                         vy_meas, thetaError_meas, omega_meas], dtype=np.float32)