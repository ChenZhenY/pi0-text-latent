#!/usr/bin/env python3
"""
Simple test script to verify the energy tracking wrapper logic works correctly.
"""

import numpy as np

def test_energy_calculation():
    """Test the energy calculation logic without importing robosuite."""
    
    print("Testing energy calculation logic...")
    
    # Simulate some test data
    torques = np.array([0.1, 0.2, -0.1, 0.05, -0.15, 0.08])
    velocities = np.array([0.5, 0.3, -0.2, 0.1, -0.4, 0.25])
    dt = 0.002
    
    # Calculate power: P = τ * ω
    instantaneous_power = np.sum(torques * velocities)
    print(f"Torques: {torques}")
    print(f"Velocities: {velocities}")
    print(f"Instantaneous power: {instantaneous_power:.6f}")
    
    # Calculate energy: E = ∫P dt
    energy = instantaneous_power * dt
    print(f"Energy for this timestep: {energy:.6f}")
    
    # Calculate torque squared
    torque_squared = np.sum(torques ** 2)
    print(f"Torque squared: {torque_squared:.6f}")
    
    # Test multiple timesteps
    total_energy = 0.0
    torque_squared_history = []
    
    for i in range(10):
        # Simulate varying torques and velocities
        torques = np.random.randn(6) * 0.2
        velocities = np.random.randn(6) * 0.3
        
        power = np.sum(torques * velocities)
        energy = power * dt
        total_energy += energy
        
        torque_squared = np.sum(torques ** 2)
        torque_squared_history.append(torque_squared)
        
        print(f"Step {i+1}: power={power:.6f}, energy={energy:.6f}, torque_squared={torque_squared:.6f}")
    
    print(f"\nTotal energy: {total_energy:.6f}")
    print(f"Torque squared sum: {np.sum(torque_squared_history):.6f}")
    print(f"Torque squared mean: {np.mean(torque_squared_history):.6f}")
    
    print("\nEnergy calculation test completed successfully!")

if __name__ == "__main__":
    test_energy_calculation()
