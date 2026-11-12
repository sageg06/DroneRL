# DroneRL
An ongoing project exploring reinforcement learning for autonomous quadcopter stabilization and indoor navigation, with the long-term goal of deploying a trained controller on a physical ESP32-S3-based drone.


## Project Overview
This is a personal learning project focused on applying reinforcement learning (RL) to autonomous drone control. The objective is to develop an RL-based control system capable of stabilizing a quadcopter, navigating indoor environments, and eventually operating on embedded hardware.

The project follows a simulation-first approach, beginning with simplified control problems before progressing toward more complex flight dynamics and real-world implementation.

Current development stage: One-dimensional (1D) vertical hovering and altitude stabilization using Proximal Policy Optimization (PPO).

## Stage 1: 1D Hover

This stage focuses on training an RL agent to maintain a target altitude by controlling the vertical thrust of a simulated drone.

A custom environment, `Hover1DEnv`, was developed using Gymnasium to model simplified vertical flight dynamics. The environment incorporates gravitational acceleration, variable drone mass, external disturbances, and simulated sensor noise.

The agent learns to regulate thrust based on its observed altitude error and vertical velocity, with the objective of maintaining a stable hover at a specified height.

Model before training:
<img width="640" height="480" alt="DroneRL_beforetraining" src="https://github.com/user-attachments/assets/daf15941-22ad-4633-aace-c9538060449a" />

Model after training:
<img width="640" height="480" alt="DroneRL_aftertraining" src="https://github.com/user-attachments/assets/573f6e75-7d64-451a-80a9-0fa75fad5ab7" />



### Simulation Parameters

| Parameter | Value |
|---|---|
| Nominal drone mass | 30 g |
| Randomized mass range | 24–36 g |
| Gravitational acceleration | 9.81 m/s² |
| Target altitude | 1.0 m |
| Simulation timestep | 0.02 s (50 Hz) |
| Maximum episode duration | 10 seconds |
| Initial altitude range | 0.2–1.8 m |
| Initial vertical velocity | −0.5 to 0.5 m/s |
| Altitude measurement noise | 5 mm standard deviation |
| Velocity measurement noise | 0.03 m/s standard deviation |
| External disturbance | ±10% of nominal drone weight |

### Reinforcement Learning Architecture

The control agent is trained using **Proximal Policy Optimization (PPO)**, implemented through Stable-Baselines3 with a multilayer perceptron (MLP) policy.

**Observation space:**

The agent receives two measurements:

- Altitude error relative to the target height
- Vertical velocity

Gaussian noise is introduced into both observations to approximate imperfect real-world sensor measurements.

**Action space:**

The agent produces a continuous control action between −1 and 1, which is mapped to an upward thrust command between zero and twice the nominal drone weight.

**Reward function:**

The current reward function encourages the agent to remain near the target altitude:

`reward = 1.0 - abs(z - target_z)`

Episodes terminate when the simulated drone exceeds the permitted altitude boundaries or reaches the maximum episode length.

### Physics Model

The simulation uses a simplified one-dimensional model of vertical motion based on Newton's second law:

`a = (T + F_disturbance) / m - g`

Where:

- `a` = vertical acceleration
- `T` = commanded thrust
- `F_disturbance` = external disturbance force
- `m` = drone mass
- `g` = gravitational acceleration

Velocity and altitude are updated at each simulation timestep using numerical integration.

To improve the controller's potential robustness, training episodes randomize initial conditions, drone mass, and external disturbances. These variations are intended to reduce dependence on a single idealized flight configuration.

## Technologies Used

- **Python** — Simulation, reinforcement learning, and evaluation
- **Gymnasium** — Custom reinforcement learning environment
- **Stable-Baselines3** — PPO implementation and policy training
- **PyTorch** — Neural network backend
- **NumPy** — Numerical calculations and simulation
- **Matplotlib** — Visualization of altitude response and controller performance

## Repository Structure

```text
DroneRL/
├── envs/
│   └── hover1d_env.py   # Custom 1D drone environment
├── train.py             # PPO agent training
├── evaluate.py          # Controller evaluation and plotting
├── check.py             # Environment compatibility validation
├── requirements.txt     # Project dependencies
├── .gitignore
└── .gitattributes
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/sageg06/DroneRL.git
cd DroneRL
```

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
pip install gymnasium stable-baselines3 numpy matplotlib
```

### Training

Run the training script:

```bash
python train.py
```

The current training configuration uses PPO with an MLP policy for 300,000 timesteps.

The trained model is saved as `hover1d_rev2_ppo.zip`.

### Evaluation

Run the evaluation script:

```bash
python evaluate.py
```

The evaluation script loads the saved PPO model and tests its performance over a simulated flight episode. It generates an altitude-versus-time plot showing the drone's vertical response relative to the target altitude.

Evaluation also supports testing under modified drone mass and external disturbance conditions.

**Note:** The current evaluation script expects the trained model file and an existing `plots/` directory for saving figures.

### Environment Validation

The custom environment can be checked for compatibility with Stable-Baselines3 using:

```bash
python check.py
```

## Long-Term Objective

The ultimate goal of DroneRL is to bridge the gap between reinforcement learning in simulation and autonomous flight on embedded hardware.

Future development will focus on expanding the current simplified flight model into a more complete quadcopter simulation, training agents for stable flight and indoor navigation, and investigating the computational and hardware constraints involved in deploying learned control policies on an ESP32-S3-based drone.

This project serves as an ongoing exploration of reinforcement learning, control theory, robotics, and embedded systems engineering.

---

**Project Status:** Early development — 1D altitude control and reinforcement learning experimentation.
