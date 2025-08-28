# Energy Tracking Wrapper

This directory contains an energy tracking wrapper for robosuite environments that tracks energy consumption, torques, and velocities throughout episodes.

## Files

- `energy_tracking_wrapper.py`: The main wrapper class that tracks energy consumption
- `test_energy_wrapper.py`: Test script to verify the energy calculation logic
- `main_lang_rollout_exp.py`: Modified main script that uses the energy tracking wrapper

## Energy Tracking Wrapper Features

The `EnergyTrackingWrapper` class provides the following functionality:

### Energy Calculation
- **Instantaneous Power**: P = τ * ω (torque × angular velocity)
- **Energy Consumption**: E = ∫P dt (integrated over time)
- **Torque Squared**: Sum of squared torques for energy analysis

### Data Collection
- Tracks torque history for each timestep
- Tracks velocity history for each timestep
- Tracks instantaneous power history
- Tracks torque squared values
- Provides episode-level energy consumption statistics

### Methods

#### `step(action)`
- Takes an action and returns (obs, reward, done, info)
- Automatically tracks energy consumption during the step
- Adds energy data to the info dictionary:
  - `energy_consumption`: Total energy consumed so far
  - `instantaneous_power`: Power for the current timestep
  - `torque_squared`: Sum of squared torques for current timestep

#### `reset_energy_tracking()`
- Resets all energy tracking data for a new episode

#### `get_energy_data()`
- Returns a dictionary with all collected energy data:
  - `total_energy`: Total energy consumption
  - `torque_history`: Array of torque values over time
  - `velocity_history`: Array of velocity values over time
  - `timestep_history`: Array of timesteps
  - `torque_squared_history`: Array of torque squared values
  - `instantaneous_power_history`: Array of power values over time

#### `get_torque_squared_sum()`
- Returns the sum of all squared torques

#### `get_torque_squared_mean()`
- Returns the mean of all squared torques

## Usage

### Basic Usage
```python
from energy_tracking_wrapper import EnergyTrackingWrapper

# Wrap your environment
env = YourRobosuiteEnv()
env = EnergyTrackingWrapper(env)

# Use normally
obs = env.reset()
for step in range(100):
    obs, reward, done, info = env.step(action)
    
    # Access energy data
    energy_consumption = info['energy_consumption']
    torque_squared = info['torque_squared']
    
    if done:
        # Get complete episode data
        energy_data = env.get_energy_data()
        print(f"Total energy: {energy_data['total_energy']:.4f} J")
        print(f"Torque squared sum: {env.get_torque_squared_sum():.4f}")
        
        # Reset for next episode
        env.reset_energy_tracking()
```

### Integration with LIBERO
The wrapper is already integrated into `main_lang_rollout_exp.py`. It will:

1. Automatically wrap the LIBERO environment
2. Log energy consumption for each episode
3. Save energy data to the results JSON files
4. Reset energy tracking between episodes

## Energy Metrics

### Energy Consumption
- **Unit**: Joules (J)
- **Calculation**: E = ∫P dt where P = τ * ω
- **Interpretation**: Total mechanical energy consumed by the robot

### Torque Squared
- **Unit**: N²⋅m²
- **Calculation**: Σ(τᵢ²) for all joints i
- **Interpretation**: Measure of control effort and energy intensity

### Instantaneous Power
- **Unit**: Watts (W)
- **Calculation**: P = τ * ω
- **Interpretation**: Rate of energy consumption at each timestep

## Testing

Run the test script to verify the energy calculation logic:
```bash
python3 test_energy_wrapper.py
```

## Notes

- The wrapper traverses through all environment wrappers to find the base MujocoEnv
- Energy tracking is reset automatically between episodes
- All energy data is available in the info dictionary returned by `step()`
- The wrapper is compatible with any robosuite environment structure
