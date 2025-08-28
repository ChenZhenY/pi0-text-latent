"""
Energy tracking wrapper for robosuite environments.
Tracks energy consumption, torques, and velocities throughout episodes.
"""

import numpy as np
from robosuite.wrappers import Wrapper


class EnergyTrackingWrapper(Wrapper):
    """
    Wrapper that tracks energy consumption, torques, and velocities.
    
    Args:
        env: The robosuite environment to wrap
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.energy_consumption = 0.0
        self.torque_history = []
        self.velocity_history = []
        self.timestep_history = []
        self.torque_squared_history = []
        self.instantaneous_power_history = []
        
    def step(self, action):
        # Get the underlying MujocoEnv (traverse through wrappers)
        mujoco_env = self._get_mujoco_env()
        
        # Store current joint velocities before step
        joint_velocities = mujoco_env.sim.data.qvel[:mujoco_env.action_dim]
        
        # Take the step
        obs, reward, done, info = self.env.step(action)
        
        # Get torques that were applied (from the most recent step)
        applied_torques = mujoco_env.sim.data.ctrl[:mujoco_env.action_dim]
        
        # Calculate instantaneous power: P = τ * ω
        instantaneous_power = np.sum(applied_torques * joint_velocities)
        
        # Calculate torque squared (for energy analysis)
        torque_squared = np.sum(applied_torques ** 2)
        
        # Accumulate energy: E = ∫P dt
        dt = mujoco_env.model_timestep
        self.energy_consumption += instantaneous_power * dt
        
        # Store history
        self.torque_history.append(applied_torques.copy())
        self.velocity_history.append(joint_velocities.copy())
        self.timestep_history.append(mujoco_env.timestep)
        self.torque_squared_history.append(torque_squared)
        self.instantaneous_power_history.append(instantaneous_power)
        
        # Add energy info to the returned info dict
        info['energy_consumption'] = self.energy_consumption
        info['instantaneous_power'] = instantaneous_power
        info['torque_squared'] = torque_squared
        
        return obs, reward, done, info
    
    def _get_mujoco_env(self):
        """Traverse through wrappers to find the base MujocoEnv"""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env
    
    def reset_energy_tracking(self):
        """Reset all energy tracking data"""
        self.energy_consumption = 0.0
        self.torque_history = []
        self.velocity_history = []
        self.timestep_history = []
        self.torque_squared_history = []
        self.instantaneous_power_history = []
    
    def get_energy_data(self):
        """Get all collected energy data"""
        return {
            'total_energy': self.energy_consumption,
            'torque_history': np.array(self.torque_history),
            'velocity_history': np.array(self.velocity_history),
            'timestep_history': self.timestep_history,
            'torque_squared_history': np.array(self.torque_squared_history),
            'instantaneous_power_history': np.array(self.instantaneous_power_history)
        }
    
    def get_torque_squared_sum(self):
        """Get the sum of squared torques (useful for energy analysis)"""
        return np.sum(self.torque_squared_history)
    
    def get_torque_squared_mean(self):
        """Get the mean of squared torques"""
        return np.mean(self.torque_squared_history) if self.torque_squared_history else 0.0
