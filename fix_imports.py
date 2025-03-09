#!/usr/bin/env python
"""
Fix import statements across the codebase.

This script updates import statements to use a consistent import format, 
either "from ember..." or "from ember..." depending on the chosen mode.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Configure which mode you want
# True: Convert all 'src.ember' to 'ember'
# False: Convert all 'ember' to 'src.ember'
CONVERT_TO_SIMPLE_IMPORTS = True

def find_python_files(root_dir: str) -> List[Path]:
    """Find all Python files in the given directory recursively."""
    return list(Path(root_dir).glob("**/*.py"))

def process_file(file_path: Path) -> Tuple[int, int]:
    """
    Process a single Python file to update imports.
    Returns tuple of (lines_changed, replacements_made)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content
    
    if CONVERT_TO_SIMPLE_IMPORTS:
        # Convert src.ember to ember
        pattern = r"from src\.ember\."
        replacement = r"from ember."
        content = re.sub(pattern, replacement, content)
        
        pattern = r"import src\.ember\."
        replacement = r"import ember."
        content = re.sub(pattern, replacement, content)
    else:
        # Convert ember to src.ember
        pattern = r"from ember\."
        replacement = r"from ember."
        content = re.sub(pattern, replacement, content)
        
        pattern = r"import ember\."
        replacement = r"import src.ember."
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        lines_changed = sum(1 for line in content.splitlines() 
                          if line != original_content.splitlines()[content.splitlines().index(line)] 
                          if content.splitlines().index(line) < len(original_content.splitlines()))
        
        replacements = len(re.findall(pattern, original_content))
        return lines_changed, replacements
    
    return 0, 0

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    
    python_files = find_python_files(root_dir)
    total_files = len(python_files)
    files_changed = 0
    total_replacements = 0
    
    print(f"Processing {total_files} Python files in {root_dir}")
    print(f"{'Converting src.ember to ember' if CONVERT_TO_SIMPLE_IMPORTS else 'Converting ember to src.ember'}")
    
    for file_path in python_files:
        lines_changed, replacements = process_file(file_path)
        if lines_changed > 0:
            files_changed += 1
            total_replacements += replacements
            print(f"Updated {file_path} - {replacements} replacements")
    
    print(f"\nSummary:")
    print(f"- Files processed: {total_files}")
    print(f"- Files changed: {files_changed}")
    print(f"- Total replacements: {total_replacements}")

if __name__ == "__main__":
    main()