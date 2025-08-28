# Energy Tracking Wrapper

This wrapper tracks energy consumption and torque statistics in robosuite environments.

## Features

- **Energy Consumption**: Tracks total energy consumption (∫P dt where P = τ * ω)
- **Torque Squared Values**: Calculates and stores torque squared for each joint
- **Instantaneous Power**: Monitors power at each timestep
- **Comprehensive History**: Stores torque and velocity history for analysis

## Usage

### Basic Usage

```python
from energy_tracking_wrapper import EnergyTrackingWrapper

# Wrap your environment
env = YourRobosuiteEnv()
env = EnergyTrackingWrapper(env)

# During episode execution, energy data is automatically tracked
obs, reward, done, info = env.step(action)

# Access energy data from info dict
energy_consumption = info['energy_consumption']
torque_squared = info['torque_squared']  # Current step
cumulative_torque_squared = info['cumulative_torque_squared']
instantaneous_power = info['instantaneous_power']
```

### Data Collection

```python
# Get comprehensive energy data
energy_data = env.get_energy_data()
print(f"Total Energy: {energy_data['total_energy']:.4f}J")

# Get torque squared statistics
torque_stats = env.get_torque_squared_stats()
print(f"Mean Torque Squared: {torque_stats['mean_torque_squared']}")
print(f"Total Torque Squared Sum: {torque_stats['total_torque_squared_sum']:.4f}")
```

### Reset Tracking

```python
# Reset energy tracking for new episode
env.reset_energy_tracking()
```

## Available Data

### Info Dict (per step)
- `energy_consumption`: Total energy consumed so far (J)
- `torque_squared`: Torque squared for current step (per joint)
- `total_torque_squared`: Sum of torque squared for current step
- `cumulative_torque_squared`: Sum of all torque squared values
- `instantaneous_power`: Power for current step (W)

### Energy Data (comprehensive)
- `total_energy`: Total energy consumption (J)
- `torque_history`: Array of all torque values
- `velocity_history`: Array of all velocity values
- `power_history`: Array of all power values
- `torque_squared_history`: Array of all torque squared values
- `mean_torque_squared`: Mean torque squared per joint
- `total_torque_squared_sum`: Sum of all torque squared values

### Torque Squared Statistics
- `mean_torque_squared`: Mean torque squared per joint
- `total_torque_squared_sum`: Sum of all torque squared values
- `max_torque_squared`: Maximum torque squared per joint
- `min_torque_squared`: Minimum torque squared per joint

## Integration with main_lang_rollout_exp.py

The wrapper is automatically applied to all environments in the main script. Energy data is:

1. **Logged during episodes**: Energy consumption and torque squared are logged for each episode
2. **Saved to files**: Episode energy data is saved as JSON files
3. **Summarized per task**: Task-level energy summaries are created

### Output Files

- `energy_data_success_{episode_idx}.json`: Energy data for successful episodes
- `energy_data_failure_{episode_idx}.json`: Energy data for failed episodes
- `task_energy_summary.json`: Summary of energy data for each task

## Testing

Run the test script to verify functionality:

```bash
python3 test_energy_tracking_simple.py
```

## Notes

- The wrapper automatically traverses through multiple wrapper layers to find the base MujocoEnv
- Energy tracking is reset at the beginning of each episode
- All data is stored in memory during episode execution and can be accessed at any time
- The wrapper is compatible with any robosuite environment structure
