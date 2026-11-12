from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

env = HoverAviary(obs=ObservationType.KIN, act=ActionType.RPM, gui=False)