#!/usr/bin/env python3
import json
import glob
import os
import numpy as np
import sys
import argparse

def analyze_energy_data(directory="."):
    """Read all energy_data*.json files in the specified directory and calculate statistics."""
    
    # Find all energy_data*.json files in the specified directory
    search_pattern = os.path.join(directory, "**/energy_data_success*.json")
    energy_files = glob.glob(search_pattern, recursive=True)
    
    if not energy_files:
        print(f"No energy_data*.json files found in {directory}!")
        return
    
    print(f"Found {len(energy_files)} energy data files in {directory}")
    
    total_energies = []
    total_torque_squared_sums = []
    
    # Read each file and extract data
    for file_path in energy_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract total_energy and total_torque_squared_sum
            if 'total_energy' in data:
                total_energies.append(data['total_energy'])
            
            if 'total_torque_squared_sum' in data:
                total_torque_squared_sums.append(data['total_torque_squared_sum'])
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Calculate statistics
    if total_energies:
        mean_total_energy = np.mean(total_energies)
        std_total_energy = np.std(total_energies)
        print(f"\nTotal Energy Statistics of success episodes:")
        print(f"  Mean: {mean_total_energy:.6f}")
        print(f"  Std:  {std_total_energy:.6f}")
        print(f"  Min:  {min(total_energies):.6f}")
        print(f"  Max:  {max(total_energies):.6f}")
        print(f"  Count: {len(total_energies)}")
    
    if total_torque_squared_sums:
        mean_torque_squared_sum = np.mean(total_torque_squared_sums)
        std_torque_squared_sum = np.std(total_torque_squared_sums)
        print(f"\nTotal Torque Squared Sum Statistics of success episodes:")
        print(f"  Mean: {mean_torque_squared_sum:.6f}")
        print(f"  Std:  {std_torque_squared_sum:.6f}")
        print(f"  Min:  {min(total_torque_squared_sums):.6f}")
        print(f"  Max:  {max(total_torque_squared_sums):.6f}")
        print(f"  Count: {len(total_torque_squared_sums)}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Directory: {directory}")
    print(f"  Files processed: {len(energy_files)}")
    print(f"  Files with total_energy: {len(total_energies)}")
    print(f"  Files with total_torque_squared_sum: {len(total_torque_squared_sums)}")

def main():
    parser = argparse.ArgumentParser(description='Analyze energy data from JSON files')
    parser.add_argument('directory', nargs='?', default='.', 
                       help='Directory to search for energy_data*.json files (default: current directory)')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist!")
        sys.exit(1)
    
    analyze_energy_data(args.directory)

if __name__ == "__main__":
    main()
