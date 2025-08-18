#!/usr/bin/env python3
"""
Test script to verify the new regex pattern
"""

import re

def test_pattern():
    content = """      (target_object_region
          (:target floor)
          (:ranges (
              (0.025 -0.125 0.07500000000000001 -0.07500000000000001)
            )
          )
      )"""
    
    pattern = r'(target_object_region\s*\(\s*:target\s+\w+\s*\(\s*:ranges\s*\(\s*\()([^)]+)(\)\s*\)\s*\)\s*\))'
    
    match = re.search(pattern, content, re.DOTALL)
    if match:
        print("Match found!")
        print(f"Prefix: '{match.group(1)}'")
        print(f"Range: '{match.group(2)}'")
        print(f"Suffix: '{match.group(3)}'")
    else:
        print("No match found")

if __name__ == "__main__":
    test_pattern()
