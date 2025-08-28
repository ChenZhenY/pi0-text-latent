import numpy as np
from robosuite.wrappers import Wrapper


class EnergyTrackingWrapper(Wrapper):
    """
    Wrapper for tracking energy consumption and torque statistics in robosuite environments.
    
    This wrapper tracks:
    - Total energy consumption (∫P dt where P = τ * ω)
    - Torque squared values for each joint
    - Instantaneous power at each timestep
    - Torque and velocity history
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.energy_consumption = 0.0
        self.torque_history = []
        self.velocity_history = []
        self.timestep_history = []
        self.power_history = []
        self.torque_squared_history = []
        
    def step(self, action, return_success_dict=False):
        # Get the underlying MujocoEnv (traverse through wrappers)
        mujoco_env = self._get_mujoco_env()
        
        # Store current joint velocities before step
        joint_velocities = mujoco_env.sim.data.qvel[:mujoco_env.action_dim]
        
        # Take the step
        obs, reward, done, info = self.env.step(action, return_success_dict=return_success_dict)
        
        # Get torques that were applied (from the most recent step)
        applied_torques = mujoco_env.sim.data.ctrl[:mujoco_env.action_dim]
        
        # Calculate instantaneous power: P = τ * ω
        instantaneous_power = np.sum(applied_torques * joint_velocities)
        
        # Calculate torque squared values for each joint
        torque_squared = applied_torques ** 2
        
        # Accumulate energy: E = ∫P dt
        dt = mujoco_env.model_timestep
        self.energy_consumption += instantaneous_power * dt
        
        # Store history
        self.torque_history.append(applied_torques.copy())
        self.velocity_history.append(joint_velocities.copy())
        self.timestep_history.append(mujoco_env.timestep)
        self.power_history.append(instantaneous_power)
        self.torque_squared_history.append(torque_squared.copy())
        
        # Add energy info to the returned info dict
        info['energy_consumption'] = self.energy_consumption
        info['instantaneous_power'] = instantaneous_power
        info['torque_squared'] = torque_squared
        info['total_torque_squared'] = np.sum(torque_squared)
        info['cumulative_torque_squared'] = self.get_total_torque_squared()
        
        return obs, reward, done, info
    
    def _get_mujoco_env(self):
        """Traverse through wrappers to find the base MujocoEnv"""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env
    
    def reset_energy_tracking(self):
        """Reset all energy tracking variables"""
        self.energy_consumption = 0.0
        self.torque_history = []
        self.velocity_history = []
        self.timestep_history = []
        self.power_history = []
        self.torque_squared_history = []
    
    def get_energy_data(self):
        """Get comprehensive energy tracking data"""
        return {
            'total_energy': self.energy_consumption,
            'torque_history': np.array(self.torque_history) if self.torque_history else np.array([]),
            'velocity_history': np.array(self.velocity_history) if self.velocity_history else np.array([]),
            'timestep_history': self.timestep_history,
            'power_history': np.array(self.power_history) if self.power_history else np.array([]),
            'torque_squared_history': np.array(self.torque_squared_history) if self.torque_squared_history else np.array([]),
            'mean_torque_squared': np.mean(self.torque_squared_history, axis=0) if self.torque_squared_history else np.array([]),
            'total_torque_squared_sum': np.sum(self.torque_squared_history) if self.torque_squared_history else 0.0
        }
    
    def get_torque_squared_stats(self):
        """Get torque squared statistics"""
        if not self.torque_squared_history:
            return {
                'mean_torque_squared': np.array([]),
                'total_torque_squared_sum': 0.0,
                'max_torque_squared': np.array([]),
                'min_torque_squared': np.array([])
            }
        
        torque_squared_array = np.array(self.torque_squared_history)
        return {
            'mean_torque_squared': np.mean(torque_squared_array, axis=0),
            'total_torque_squared_sum': np.sum(torque_squared_array),
            'max_torque_squared': np.max(torque_squared_array, axis=0),
            'min_torque_squared': np.min(torque_squared_array, axis=0)
        }
    
    def get_current_torque_squared(self):
        """Get the current torque squared values (most recent step)"""
        if not self.torque_squared_history:
            return np.array([])
        return self.torque_squared_history[-1]
    
    def get_total_torque_squared(self):
        """Get the sum of all torque squared values so far"""
        if not self.torque_squared_history:
            return 0.0
        return np.sum(self.torque_squared_history)
