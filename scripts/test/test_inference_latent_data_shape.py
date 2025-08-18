#!/usr/bin/env python3
"""
Test script to verify the structure of inference latent collection data
"""

import pickle
import pathlib
import numpy as np

def print_dict_structure(d, indent=0):
    """Recursively print dictionary structure with array shapes."""
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_dict_structure(value, indent + 1)
        elif isinstance(value, np.ndarray):
            print("  " * indent + f"{key}: numpy array with shape {value.shape}")
        elif isinstance(value, list):
            print("  " * indent + f"{key}: list with length {len(value)}")
        else:
            print("  " * indent + f"{key}: {type(value)}")

def test_inference_latent_structure(pkl_path):
    """Test function to examine inference latent collection data structure."""
    
    # Load pickle file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("\nData structure:")
    print("==============")
    print_dict_structure(data)

if __name__ == "__main__":
    # Example usage
    # pkl_path = pathlib.Path("data/inference_latents/task_0_pick_up_the_block.pkl")
    pkl_path = "/research/data/zhenyang/pi0-text-latent/data/inference_latents/task_0_pick_up_the_alphabet_soup_and_place_it_in_the_basket.pkl"
    test_inference_latent_structure(pkl_path)
