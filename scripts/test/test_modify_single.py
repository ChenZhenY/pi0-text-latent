#!/usr/bin/env python3
"""
Test script to modify a single BDDL file and see the changes
"""

import re
from pathlib import Path

def parse_range_string(range_str):
    """Parse a range string like '(0.025 -0.125 0.07500000000000001 -0.07500000000000001)'"""
    # Remove parentheses and split by spaces
    numbers = range_str.strip('()').split()
    if len(numbers) == 4:
        return [float(x) for x in numbers]
    return None

def expand_range_to_15cm(range_coords):
    """Expand a range to have 15cm (0.15m) in both directions, centered on current center"""
    if len(range_coords) != 4:
        return range_coords
    
    x_min, y_min, x_max, y_max = range_coords
    
    # Calculate current center
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # Expand by 15cm (0.15m) in both directions
    half_expansion = 0.15 / 2  # 7.5cm in each direction
    
    new_x_min = center_x - half_expansion
    new_x_max = center_x + half_expansion
    new_y_min = center_y - half_expansion
    new_y_max = center_y + half_expansion
    
    return [new_x_min, new_y_min, new_x_max, new_y_max]

def format_range_string(range_coords):
    """Format range coordinates back to string format"""
    return f"({range_coords[0]} {range_coords[1]} {range_coords[2]} {range_coords[3]})"

def test_modify_file():
    input_file = "third_party/modified_libero/libero/libero/bddl_files/libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl"
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    print("Original content:")
    print(content)
    print("\n" + "="*50 + "\n")
    
    # Try different regex patterns
    patterns = [
        r'target_object_region\s*\(\s*:target\s+\w+\s*\(\s*:ranges\s*\(\s*\(([^)]+)\)\s*\)\s*\)\s*\)',
        r'target_object_region\s*\(\s*:target\s+\w+\s*\(\s*:ranges\s*\(\s*\(([^)]+)\)\s*\)\s*\)\s*\)',
        r'target_object_region.*?\(:ranges\s*\(\s*\(([^)]+)\)\s*\)\s*\)\s*\)',
    ]
    
    for i, pattern in enumerate(patterns):
        print(f"Testing pattern {i+1}: {pattern}")
        matches = re.findall(pattern, content, re.DOTALL)
        print(f"Matches found: {matches}")
        if matches:
            for match in matches:
                print(f"  Range: {match}")
                range_coords = parse_range_string(match)
                if range_coords:
                    expanded = expand_range_to_15cm(range_coords)
                    print(f"  Original: {range_coords}")
                    print(f"  Expanded: {expanded}")
        print()

if __name__ == "__main__":
    test_modify_file()
