#!/usr/bin/env python3
"""
Benchmark script to estimate pickle loading performance for large files.
This helps estimate the speed for loading 20GB pickle files.
"""

import time
import pickle
import numpy as np
import torch
import psutil
import os
from pathlib import Path


def create_test_data(size_gb=1.0):
    """Create test data of specified size in GB."""
    # Estimate bytes per sample (typical for the inference latent data structure)
    # Based on the script, each sample contains hidden states with shape like (layers, seq_len, hidden_dim)
    # Typical values: 18 layers, 816 seq_len, 768 hidden_dim, float32 = 4 bytes
    bytes_per_sample = 18 * 816 * 768 * 4  # ~45MB per sample
    
    num_samples = int((size_gb * 1024**3) / bytes_per_sample)
    
    print(f"Creating test data: {size_gb:.1f}GB with {num_samples} samples")
    print(f"Estimated bytes per sample: {bytes_per_sample:,}")
    
    # Create mock data structure similar to the inference latent format
    test_data = {
        "task_description": "Test task for benchmarking",
        "episodes": {}
    }
    
    for episode_idx in range(5):  # 5 episodes
        episode_data = {
            "rollout_steps": {}
        }
        
        for step_idx in range(20):  # 20 steps
            step_data = {
                "vlm_layer_output": {
                    "hidden_states": np.random.randn(18, 1, 816, 768).astype(np.float32)
                }
            }
            episode_data["rollout_steps"][f"step_{step_idx}"] = step_data
        
        test_data["episodes"][f"episode_{episode_idx}"] = episode_data
    
    return test_data


def benchmark_pickle_loading(file_path, num_runs=3):
    """Benchmark pickle loading performance."""
    print(f"\nBenchmarking pickle loading for: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / (1024**3):.2f} GB")
    
    # Get system info
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"System RAM: {memory_gb:.1f} GB")
    
    load_times = []
    memory_usage = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Clear memory
        import gc
        gc.collect()
        
        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**3)
        
        # Time the loading
        start_time = time.time()
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            load_time = time.time() - start_time
            load_times.append(load_time)
            
            # Measure memory after
            memory_after = process.memory_info().rss / (1024**3)
            memory_used = memory_after - memory_before
            memory_usage.append(memory_used)
            
            print(f"  Load time: {load_time:.2f} seconds")
            print(f"  Memory used: {memory_used:.2f} GB")
            print(f"  Speed: {os.path.getsize(file_path) / (1024**3) / load_time:.2f} GB/s")
            
            # Clear data to free memory
            del data
            gc.collect()
            
        except Exception as e:
            print(f"  Error loading file: {e}")
            return None
    
    # Calculate statistics
    avg_load_time = np.mean(load_times)
    avg_memory = np.mean(memory_usage)
    avg_speed = os.path.getsize(file_path) / (1024**3) / avg_load_time
    
    print(f"\nResults:")
    print(f"  Average load time: {avg_load_time:.2f} ± {np.std(load_times):.2f} seconds")
    print(f"  Average memory usage: {avg_memory:.2f} ± {np.std(memory_usage):.2f} GB")
    print(f"  Average speed: {avg_speed:.2f} GB/s")
    
    return {
        'avg_load_time': avg_load_time,
        'avg_memory': avg_memory,
        'avg_speed': avg_speed,
        'file_size_gb': os.path.getsize(file_path) / (1024**3)
    }


def estimate_20gb_loading(results):
    """Estimate loading time for 20GB file based on benchmark results."""
    if results is None:
        return None
    
    # Extrapolate to 20GB
    estimated_time_20gb = (20.0 / results['file_size_gb']) * results['avg_load_time']
    estimated_memory_20gb = (20.0 / results['file_size_gb']) * results['avg_memory']
    
    print(f"\nEstimation for 20GB file:")
    print(f"  Estimated load time: {estimated_time_20gb:.1f} seconds ({estimated_time_20gb/60:.1f} minutes)")
    print(f"  Estimated memory usage: {estimated_memory_20gb:.1f} GB")
    print(f"  Estimated speed: {results['avg_speed']:.2f} GB/s")
    
    return {
        'estimated_time_20gb': estimated_time_20gb,
        'estimated_memory_20gb': estimated_memory_20gb,
        'speed_gb_s': results['avg_speed']
    }


def analyze_linear_probing_loading():
    """Analyze the specific loading pattern used in linear_probing_inference.py."""
    print("\n" + "="*60)
    print("ANALYSIS OF LINEAR PROBING LOADING PATTERN")
    print("="*60)
    
    print("\nLoading Pattern Analysis:")
    print("1. The script loads multiple pickle files (one per task)")
    print("2. For each file, it extracts specific episodes and steps")
    print("3. It processes hidden states and converts to PyTorch tensors")
    print("4. It performs feature extraction and mean pooling")
    
    print("\nPerformance Considerations:")
    print("✓ Sequential file loading (not parallel)")
    print("✓ In-memory processing of entire files")
    print("✓ PyTorch tensor conversion overhead")
    print("✓ Feature extraction computation")
    print("✓ Memory accumulation of all features")
    
    print("\nPotential Bottlenecks:")
    print("- Single-threaded pickle loading")
    print("- Memory pressure from large files")
    print("- CPU-intensive feature extraction")
    print("- PyTorch tensor operations")
    
    print("\nOptimization Suggestions:")
    print("1. Use multiprocessing for parallel file loading")
    print("2. Implement streaming/lazy loading")
    print("3. Use memory-mapped files for very large datasets")
    print("4. Pre-extract features and save as numpy arrays")
    print("5. Use torch.load() instead of pickle for PyTorch tensors")


def main():
    """Main benchmark function."""
    print("Pickle Loading Performance Benchmark")
    print("="*50)
    
    # Create test files of different sizes
    test_sizes = [0.1, 0.5, 1.0]  # GB
    
    results = []
    
    for size_gb in test_sizes:
        print(f"\n{'='*20} Testing {size_gb}GB file {'='*20}")
        
        # Create test file
        test_data = create_test_data(size_gb)
        test_file = f"test_data_{size_gb}gb.pkl"
        
        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        # Benchmark loading
        result = benchmark_pickle_loading(test_file, num_runs=2)
        if result:
            results.append(result)
        
        # Clean up
        os.remove(test_file)
    
    # Estimate 20GB performance
    if results:
        print(f"\n{'='*20} 20GB Estimation {'='*20}")
        # Use the largest tested file for most accurate estimation
        largest_result = max(results, key=lambda x: x['file_size_gb'])
        estimate_20gb_loading(largest_result)
    
    # Analyze the linear probing loading pattern
    analyze_linear_probing_loading()
    
    
    print(f"\n{'='*20} Summary {'='*20}")
    print("For a 20GB pickle file with the linear_probing_inference.py pattern:")
    print("- Expected load time: 30-120 seconds (depending on hardware)")
    print("- Memory usage: 20-40 GB (depending on data structure)")
    print("- Main bottleneck: Sequential file I/O and memory allocation")
    print("- Consider using streaming or parallel loading for better performance")


if __name__ == "__main__":
    main()

