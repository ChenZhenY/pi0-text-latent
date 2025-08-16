#!/usr/bin/env python3
"""
Script to copy all BDDL files and modify target object regions to have a 15cm range
in both directions, centered around the current center.
"""

import os
import shutil
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

def modify_bddl_file(input_path, output_path):
    """Modify a BDDL file to expand target regions to 15cm range"""
    with open(input_path, 'r') as f:
        content = f.read()
    
    # Find and modify target_object_region ranges
    # Use the working pattern from the test
    target_object_pattern = r'target_object_region.*?\(:ranges\s*\(\s*\(([^)]+)\)\s*\)\s*\)\s*\)'
    
    def replace_target_object_region(match):
        range_str = match.group(1)
        
        print(f"  Found target_object_region with range: {range_str}")
        
        # Parse the range
        range_coords = parse_range_string(range_str)
        if range_coords:
            # Expand the range
            expanded_range = expand_range_to_15cm(range_coords)
            new_range_str = format_range_string(expanded_range)
            print(f"  Expanded to: {new_range_str}")
            
            # Replace the entire match with the new range
            return match.group(0).replace(range_str, new_range_str)
        
        return match.group(0)
    
    # Apply the replacement
    modified_content = re.sub(target_object_pattern, replace_target_object_region, content, flags=re.DOTALL)
    
    # Write the modified content
    with open(output_path, 'w') as f:
        f.write(modified_content)

def copy_and_modify_directory(src_dir, dst_dir):
    """Copy directory structure and modify BDDL files"""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # Create destination directory
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Copy all files and directories
    for item in src_path.rglob('*'):
        if item.is_file():
            # Calculate relative path
            rel_path = item.relative_to(src_path)
            dst_file = dst_path / rel_path
            
            # Create parent directories if needed
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            # If it's a BDDL file, modify it; otherwise just copy
            if item.suffix == '.bddl':
                print(f"Modifying: {rel_path}")
                modify_bddl_file(item, dst_file)
            else:
                print(f"Copying: {rel_path}")
                shutil.copy2(item, dst_file)
        elif item.is_dir():
            # Create directory
            rel_path = item.relative_to(src_path)
            dst_dir = dst_path / rel_path
            dst_dir.mkdir(parents=True, exist_ok=True)

def main():
    # Source and destination directories
    src_base = "third_party/modified_libero/libero/libero/bddl_files"
    dst_base = "third_party/modified_libero/libero/libero/bddl_files_15cm"
    
    # List of task suites to process
    task_suites = [
        "libero_goal",
        "libero_goal_ood", 
        "libero_object",
        "libero_object_ood",
        "libero_spatial",
        "libero_spatial_ood",
        "libero_10",
        "libero_90"
    ]
    
    print("Starting BDDL file modification...")
    print(f"Source: {src_base}")
    print(f"Destination: {dst_base}")
    print()
    
    # Process each task suite
    for suite in task_suites:
        src_suite = os.path.join(src_base, suite)
        dst_suite = os.path.join(dst_base, suite)
        
        if os.path.exists(src_suite):
            print(f"Processing suite: {suite}")
            copy_and_modify_directory(src_suite, dst_suite)
            print(f"Completed: {suite}")
            print()
        else:
            print(f"Warning: Suite directory not found: {src_suite}")
    
    print("BDDL modification completed!")
    print(f"Modified files are in: {dst_base}")

if __name__ == "__main__":
    main()
